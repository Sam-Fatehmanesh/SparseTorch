import torch
from torch import nn
from sparsetorch.cortex.cortexutil import *
from sparsetorch.sparse.sparselinear import SparseLinear

def temp_optimizer():
    pass

#neurons which can locally interact and learn
class NeuronCluster(nn.Module):
    def __init__(self, in_features, out_features, dendrites_per_axon = 8, act_func = nn.functional.relu, BPoptimizer = torch.optim.SGD, sparse_dendrites=True):
        super(CortexLayer, self).__init__(in_features, out_features)
        self.neuron_count = out_features

        self.synapse_density = None # 1 = fully connected, 0.5 = .5 sparse connectivity

        if sparse_dendrites:
            self.dendrites = SparseLinear(in_features, dendrites_per_axon*self.out_features, grad_enabled)
        else:
            self.dendrites = nn.Linear(in_features, dendrites_per_axon*self.out_features)
        self.axons = nn.Conv1d(1, 1, kernel_size=dendrites_per_axon, stride=dendrites_per_axon)
        self.opt = torch.optim.SGD(self.parameters(), lr=0.01)

    def forward(self, x):
        x = act_func(self.dendrites(x))
        # this needs to be made batch spesific
        x = nn.functional.normalize(act_func(self.axons(x)))
        return x

    # rewireing + weight optimizing
    def forward_learn(self, x, signal, optimizer):
        # weight updates
        out = self.forward(x)
        sigout = self.forward(signal)
        # Perhaps add L1 reg
        loss = torch.sum(nn.functional.cosine_similarity(out, sigout))

        self.opt.zero_grad()
        loss.backwards()
        self.opt.step()

        # Evolve Connectivity
        if type(self.dendrites) == SparseLinear:
            
            pass

        return out, sigout
        

    # How to allow NC to find relationships between all inputs while makeing unique findings: minimize cosign similarity between multiple NCs?