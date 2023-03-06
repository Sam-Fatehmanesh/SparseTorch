import torch
from torch import sparse
from torch import nn
from torch import Tensor
from sparsetorch.util import *

# matrix multiplies a and b together and then adds term to the product
def batch_addmm(matrix, vector_batch, term):
    batch_size = vector_batch.shape[0]
    # Stack the vector batch into columns. (b, n, 1) -> (n, b)
    vectors = vector_batch.transpose(0, 1).reshape(-1, batch_size)

    # A matrix-matrix product is a batched matrix-vector product of the columns.
    # And then reverse the reshaping. (m, b) -> (b, m, 1)
    term = torch.tile(term, (batch_size,1)).transpose(0, 1)
    return sparse.addmm(term, matrix, vectors).transpose(0, 1)