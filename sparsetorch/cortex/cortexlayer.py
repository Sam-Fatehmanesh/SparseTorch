import torch
from torch import nn
from sparsetorch.cortex.cortexutil import *
from sparsetorch.cortex.neruoncluster import NeuronCluster

class CortexLayer():
    def __init__(self, in_features, out_features, cluster_count, dendrites_per_axon):
        assert cluster_count <= out_features, "cluster_count must be less than or equal to out_features"
        assert out_features % cluster_count == 0, "out_features must be divisible by cluster_count"
        self.neuron_clusters = [NeuronCluster(in_features=in_features, out_features=out_features/cluster_count, dendrites_per_axon=dendrites_per_axon) for i in range(cluster_count)]
            

    def forward(self, x):
        cluster_outputs = [None] * len(self.neuron_clusters)
        for i in range(len(self.neuron_clusters)):
            cluster_outputs[i] = self.neuron_clusters[i].forward(x)
        return torch.cat(tuple(cluster_outputs))


    # def forward_learn(self, x):
    #     cluster_outputs = [None] * len(self.neuron_clusters)

    #     for i in range(len(self.neuron_clusters)):
    #         cluster_outputs[i] = self.neuron_clusters[i].forward_learn(x)

    #     return torch.cat(tuple(cluster_outputs))