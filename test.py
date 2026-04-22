from codon.exp.block.bio import EpisodicAssociativeBlock

import torch

eab = EpisodicAssociativeBlock(2, 4096, 500)

print(eab.count_params(human_readable=True))

print(eab.stats)

print(eab.auto_memorize(0, 10, torch.Tensor([0.5, 0.4])))
print(eab.memorize(torch.Tensor([0.5, 0.346]), 1))

print(eab.associate(torch.Tensor([0.5, 0.3]), topk=5))
