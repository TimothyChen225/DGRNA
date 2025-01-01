# DGRNA: RNA foundation model with bidirectional mamba2-attention

![image](https://github.com/TimothyChen225/DGRNA/blob/main/picture/DGRNA.png)

## About

In recent years, the rapid development of sequencing technology has significantly enhanced our understanding of RNA biology and advanced RNA-based therapies, resulting in a huge volume of RNA data. Current RNA large language models are primarily based on Transformer architecture, which cannot efficiently process long RNA sequences, while the Mamba architecture can effectively alleviates the quadratic complexity associated with Transformers. Based on this, we propose a large foundational model DGRNA based on the bidirectional Mamba trained on 100 million RNA sequences, which has demonstrated exceptional performance across multiple downstream tasks compared to existing RNA language models.

## Installation

- **triton                    2.2.0**
- **causal-conv1d    1.2.2.post1**
- **mamba-ssm        2.2.2**
- **fairseq                  0.12.2**

## weights

Download the [pretraining  file](https://drive.google.com/drive/folders/1LQOIo-fvij3L2dPEA2zfyOGNn3mkz4KE?usp=sharing) and place it in~/. cache/torch/hub/checkpoints/

## Usage

```python
import DGRNA
import torch

device = torch.device("cpu")#.device("cuda:2")
model, alphabet, _ = DGRNA.mamba2_pretrained.rna_mamba2_L24()
model = model.to(device)

batch_converter = alphabet.get_batch_converter()
model.eval()


data = [
    ("RNA1", "GGGUGCGAUCAUACCAGCACUAAUGCCCUCCUGGGAAGUCCUCGUGUUGCACCCCU"),
    ("RNA2", "GGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCA"),
    ("RNA3", "CGAUUCNCGUUCCC--CCGCCUCCA"),
]
batch_labels, batch_strs, batch_tokens = batch_converter(data)
batch_tokens = batch_tokens.to(device)
with torch.no_grad():
    #with torch.cuda.device(batch_tokens.device):
        results = model(batch_tokens)
        print(results)
```

## Citation

```
@article {Yuan2024.10.31.621427,
	author = {Yuan, Ye and Chen, Qushuo and Pan, Xiaoyong},
	title = {DGRNA: a long-context RNA foundation model with bidirectional attention Mamba2},
	year = {2024},
	journal = {bioRxiv}
}

```

The model of this code is built based on the [ESM](https://github.com/facebookresearch/esm) and [Fairseq](https://github.com/facebookresearch/fairseq) third party library. We greatly appreciate these two excellent works!

