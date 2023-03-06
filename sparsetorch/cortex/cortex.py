import torch
from torch import nn
from sparsetorch.cortex.cortexlayer import CortexLayer

class Cortex(nn.Module):
    def __init__(self):
        super(Cortex, self).__init__()
        