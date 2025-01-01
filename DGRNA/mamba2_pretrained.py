# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json
import math
import os
import re
import warnings
from pathlib import Path
from functools import partial
import torch

import torch.nn as nn

import copy

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.block import Block
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
from .multihead_attention_mha import MultiheadAttention
# from .esm2 import ESM2, MAMBA2
# from .mamba2 import MambaLMHeadModel
import DGRNA.data as DGdata


def create_block(
    d_model,
    d_intermediate,
    ssm_cfg=None,
    attn_layer_idx=None,
    attn_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if layer_idx not in attn_layer_idx:
        # Create a copy of the config to modify
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        ssm_layer = "Mamba2"#ssm_cfg.pop("layer", "Mamba1") #
        if ssm_layer not in ["Mamba1", "Mamba2"]:
            raise ValueError(f"Invalid ssm_layer: {ssm_layer}, only support Mamba1 and Mamba2")
        mixer_cls = partial(
            Mamba2 if ssm_layer == "Mamba2" else Mamba,
            layer_idx=layer_idx,
            **ssm_cfg,
            **factory_kwargs
        )
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
        )
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int,
        vocab_size: int,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, inference_params=None, **mixer_kwargs):
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )
        return hidden_states


class BiDirectionMixerModel(nn.Module):
    """
    ref to https://github.com/programmablebio/ptm-mamba/blob/main/protein_lm/modeling/models/mamba/lm.py
    """
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int,
        vocab_size: int,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)
        self.gate = nn.Linear(d_model, 1, )
        #self.gmlp = GatedMLP(in_features=d_model, out_features=d_model)
        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.forward_layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )
        self.backward_layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )
        
        self.attn_layers = nn.ModuleList(
                [
                    MultiheadAttention(
                        d_model,
                        32,
                        use_flash_attn=True,
                        return_residual=False,
                        rotary_emb_dim=(d_model//32),
                        layer_idx=i
                    ) for i in range(1)
                ])

        self.hidden_fc = nn.ModuleList(
            [nn.Linear( 2* d_model, d_model) for i in range(n_layer)]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, inference_params=None,embedding=None, **mixer_kwargs):
        hidden_states = self.embedding(input_ids)
        #hidden_states_r = self.attn_layers[0](hidden_states)
        # embedding = torch.zeros_like(hidden_states) if embedding is None else embedding # BxLxD
        # gate = self.gate(torch.cat([hidden_states, embedding], dim=-1)).sigmoid()
        # contact = torch.einsum("bled, blce->blcd", hidden_states.unsqueeze(-2) , hidden_states.unsqueeze(-1))
        # contact = self.gate(contact).sigmoid().squeeze(-1) # BxLxD
        # hidden_states = hidden_states + contact# + embedding * (1 - gate)
        residual = None
        for f_layer, b_layer, h_fc in zip(
                self.forward_layers, self.backward_layers, self.hidden_fc
        ):
            hidden_states_f, residual_f = f_layer(
                hidden_states, residual, inference_params=inference_params
            )
            flip_residual = residual.flip([1]) if residual is not None else None
            hidden_states_b, residual_b = b_layer(
                hidden_states.flip([1]), flip_residual, inference_params=inference_params
            )
            hidden_states = h_fc(torch.cat([hidden_states_f, hidden_states_b.flip([1])], dim=-1)) + hidden_states_f
            #hidden_states = gates*hidden_states_f + (1-gates)*hidden_states_b.flip([1])
            residual = 0.5 * (residual_f + residual_b.flip([1]))
        #hidden_states = self.norm_f(self.gmlp(hidden_states))
        hidden_states = self.attn_layers[0](hidden_states)# + hidden_states_r

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )
        return hidden_states


class MambaLMHeadModel(nn.Module):

    def __init__(
        self,
        config,#: MambaConfig
        dictionary,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        d_intermediate = config.d_intermediate
        vocab_size = len(dictionary)
        ssm_cfg = config.ssm_cfg
        attn_layer_idx = config.attn_layer_idx
        attn_cfg = config.attn_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        # self.backbone = MixerModel(
        #     d_model=d_model,
        #     n_layer=n_layer,
        #     d_intermediate=d_intermediate,
        #     vocab_size=vocab_size,
        #     ssm_cfg=ssm_cfg,
        #     attn_layer_idx=attn_layer_idx,
        #     attn_cfg=attn_cfg,
        #     rms_norm=rms_norm,
        #     initializer_cfg=initializer_cfg,
        #     fused_add_norm=fused_add_norm,
        #     residual_in_fp32=residual_in_fp32,
        #     **factory_kwargs,
        # )
        self.backbone = BiDirectionMixerModel(
            d_model=d_model,
            n_layer=n_layer,
            d_intermediate=d_intermediate,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        )
        # self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)
        self.lm_head = Mamba2LMHead(d_model, vocab_size,activation_fn=config.activation_fn, weight=self.backbone.embedding.weight)
        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        if self.config.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0, masked_tokens=None, **mixer_kwargs):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.backbone(input_ids, inference_params=inference_params, **mixer_kwargs)

        # if num_last_tokens > 0:
        #     hidden_states = hidden_states[:, -num_last_tokens:]
        # lm_logits = self.lm_head(hidden_states, masked_tokens)
        lm_logits = self.lm_head(hidden_states)
        # CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return hidden_states#CausalLMOutput(logits=lm_logits)

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=4)

class Mamba2LMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = nn.functional.gelu#utils.get_activation_fn(activation_fn)
        self.layer_norm = nn.LayerNorm(embed_dim)
        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation

        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = nn.functional.linear(x, self.weight) + self.bias

        return x



def _has_regression_weights(model_name):
    """Return whether we expect / require regression weights;
    Right now that is all models except ESM-1v, ESM-IF, and partially trained ESM2 models"""
    return not ("esm1v" in model_name or "esm_if" in model_name or "270K" in model_name or "500K" in model_name)


def load_model_and_alphabet(model_name):
    if model_name.endswith(".pt"):  # treat as filepath
        return load_model_and_alphabet_local(model_name)
    else:
        return load_model_and_alphabet_hub(model_name)


def load_hub_workaround(url, download_name=None):
    try:
        data = torch.hub.load_state_dict_from_url(url, progress=True, map_location='cpu', file_name=download_name)
    except RuntimeError:
        # Pytorch version issue - see https://github.com/pytorch/pytorch/issues/43106
        if download_name == None:
            fn = Path(url).name
        else:
            fn = download_name
        data = torch.load(
            f"{torch.hub.get_dir()}/checkpoints/{fn}",
            map_location="cpu",
        )
    return data


def load_regression_hub(model_name):
    url = f"https://dl.fbaipublicfiles.com/fair-esm/regression/{model_name}-contact-regression.pt"
    regression_data = load_hub_workaround(url)
    return regression_data


def _download_model_and_regression_data(model_name):
    url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt"
    model_data = load_hub_workaround(url)
    if _has_regression_weights(model_name):
        regression_data = load_regression_hub(model_name)
    else:
        regression_data = None
    return model_data, regression_data


def load_model_and_alphabet_hub(model_name):
    model_data, regression_data = _download_model_and_regression_data(model_name)
    return load_model_and_alphabet_core(model_name, model_data, regression_data)


def load_model_and_alphabet_local(model_location):
    """Load from local path. The regression weights need to be co-located"""
    model_location = Path(model_location)
    model_data = torch.load(str(model_location), map_location="cpu")
    model_name = model_location.stem
    if _has_regression_weights(model_name):
        regression_location = str(model_location.with_suffix("")) + "-contact-regression.pt"
        regression_data = torch.load(regression_location, map_location="cpu")
    else:
        regression_data = None
    return load_model_and_alphabet_core(model_name, model_data, regression_data)


def has_emb_layer_norm_before(model_state):
    """Determine whether layer norm needs to be applied before the encoder"""
    return any(k.startswith("emb_layer_norm_before") for k, param in model_state.items())


def _load_model_and_alphabet_core_v2(model_data):
    def upgrade_state_dict(state_dict):
        """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
        prefixes = ["encoder.sentence_encoder.", "encoder."]
        pattern = re.compile("^" + "|".join(prefixes))
        state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
        return state_dict

    cfg = model_data["cfg"]["model"]
    state_dict = model_data["model"]
    # print("state_dict: ",state_dict)
    state_dict = upgrade_state_dict(state_dict)
    alphabet = DGdata.Alphabet.from_architecture("ESM-1b")
    # print("alphabet: ",len(alphabet))
    # model = ESM2(
    #     num_layers=cfg.encoder_layers,
    #     embed_dim=cfg.encoder_embed_dim,
    #     attention_heads=cfg.encoder_attention_heads,
    #     alphabet=alphabet,
    #     token_dropout=True, #cfg.token_dropout
    # )
    # model = MAMBA2(
    #     num_layers=cfg.encoder_layers,
    #     embed_dim=cfg.encoder_embed_dim,
    #     attention_heads=cfg.encoder_attention_heads,
    #     alphabet=alphabet,
    #     token_dropout=True, #cfg.token_dropout
    # )
    model = MambaLMHeadModel(cfg, alphabet)

    return model, alphabet, state_dict


def load_model_and_alphabet_core(model_name, model_data, regression_data=None):
    # print("regression_data: ",regression_data)
    if regression_data is not None:
        model_data["model"].update(regression_data["model"])
    # print(model_data)
    # exit(0)
    #print(model_data["cfg"])
    # pra = lambda s: ''.join(s.split('encoder_')[1:] if 'encoder' in s else s)
    # model_args = {pra(arg[0]): arg[1] for arg in (model_data["cfg"]).items()}
    # model_args = Namespace(**model_args)
    # print(model_data['cfg']["model"])
    # exit(0)
    model_args = model_data['cfg']["model"]

    model, alphabet, model_state = _load_model_and_alphabet_core_v2(model_data)


    expected_keys = set(model.state_dict().keys())
    found_keys = set(model_state.keys())
    # print("alphabet.eos_idx: ",alphabet.eos_idx)
    # if regression_data is None:
    #     expected_missing = {"contact_head.regression.weight", "contact_head.regression.bias"}
    #     error_msgs = []
    #     missing = (expected_keys - found_keys) - expected_missing
    #
    #     if missing:
    #         error_msgs.append(f"Missing key(s) in state_dict: {missing}.")
    #     unexpected = found_keys - expected_keys
    #
    #     if unexpected:
    #         error_msgs.append(f"Unexpected key(s) in state_dict: {unexpected}.")
    #
    #     if error_msgs:
    #         raise RuntimeError(
    #             "Error(s) in loading state_dict for {}:\n\t{}".format(
    #                 model.__class__.__name__, "\n\t".join(error_msgs)
    #             )
    #         )
    #     if expected_missing - found_keys:
    #         warnings.warn(
    #             "Regression weights not found, predicting contacts will not produce correct results."
    #         )
    expected_missing = {"contact_head.regression.weight", "contact_head.regression.bias"}
    error_msgs = []
    missing = (expected_keys - found_keys) - expected_missing
    unexpected = found_keys - expected_keys

    if unexpected:
        error_msgs.append(f"Unexpected key(s) in state_dict: {unexpected}.")

    if error_msgs:
        raise RuntimeError(
            "Error(s) in loading state_dict for {}:\n\t{}".format(
                model.__class__.__name__, "\n\t".join(error_msgs)
            )
        )
    if missing:
        error_msgs.append(f"Missing key(s) in state_dict: {missing}.")

    # model.load_state_dict(model_state, strict=regression_data is not None)
    model.load_state_dict(model_state)
    return model, alphabet, model_args


def load_mamba2_model_and_alphabet_hub(model_name):
    if model_name == "rna_fm_t12":
        url = f"https://proj.cse.cuhk.edu.hk/rnafm/api/download?filename=checkpoint_best_100M.pt"
        model_data = load_hub_workaround(url, download_name="checkpoint_best_100M.pt")
        #url = f"https://proj.cse.cuhk.edu.hk/rnafm/api/download?filename=RNA-FM_SS-ResNet.pth"
        #model_data = load_hub_workaround(url, download_name="RNA-FM_SS-ResNet.pth")
        regression_data = None

    else:
        raise Exception("Unknown model name: {}".format(model_name))
    return load_model_and_alphabet_core(model_name, model_data, regression_data)


def rna_mamba2_L24(model_location=None):
    # if model_location is not None and os.path.exists(model_location):
    #     # local
    #     return load_model_and_alphabet_local(model_location, theme="rna")  # "./pretrained/RNA-FM_pretrained.pth"
    # else:
        return load_mamba2_model_and_alphabet_hub("rna_fm_t12")



