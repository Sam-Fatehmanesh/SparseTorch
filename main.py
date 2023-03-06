import torch
from torch import nn
from sparsetorch.sparse.sparseutil import *
from tests.BPtraintests import *
from tests.optimtests import *
from sparsetorch.sparse.sparselinear import SparseLinear
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import time


print(torch.version.__version__)


sparseavg = 0

rounds = 200

for i in range(rounds):
    sparseavg += testBPTrainMNIST(True)
sparseavg = sparseavg / 200.0


denseavg = 0

for i in range(rounds):
    denseavg += testBPTrainMNIST(False)
denseavg = denseavg / 200.0

print(sparseavg)
print(denseavg)