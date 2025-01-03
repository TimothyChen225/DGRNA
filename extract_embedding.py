import DGRNA
import torch

device = torch.device("cuda:2")
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
