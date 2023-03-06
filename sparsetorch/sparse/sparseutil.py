import random
import torch
import warnings

def randSparseIndices(shape, sparsity, attempts, device) -> torch.Tensor:
    # Total number of enteries    
    nz = 1.0
    for dim in shape:
        nz *= dim
    # Total number of nonzero entries
    nnz = round(nz * (1.0 - sparsity))
    # Index of values in each dimention
    indices = []
    duplicate_buffer_factor = 1

    best_indices = torch.zeros(1)

    for i in range(attempts):
        indices = []
        for dim in shape:
            index = torch.randint(0, dim, (round(nnz*duplicate_buffer_factor),), device=device)
            indices.append(index)
        indices = torch.vstack(indices)
        indices = torch.transpose(indices, 0, 1)

        if attempts > 1:
            indices = indices.unique(dim=0)

        #print(indices.size()[0])
        #print(best_indices.size()[0])

        if indices.size()[0] > best_indices.size()[0]:
            best_indices = indices

        if nnz <= indices.size()[0]:
            break

    indices = best_indices

    if nnz < indices.size()[0]:
        indices = indices[:nnz]

    
    indices = torch.transpose(indices, 0, 1)

    # padding in order to preserve tensor shape and keep pytorch happy
    for i in range(len(shape)):
        indices[i, indices.size()[1] - 1] = shape[i] - 1
    
    return indices.float()

def randCOOTensor(shape, sparsity=0.9, attempts=1, dtype=None, device=None) -> torch.Tensor:
    # Total number of enteries    
    nz = 1.0
    for dim in shape:
        nz *= dim
    # Total number of nonzero entries
    nnz = round(nz * (1.0 - sparsity))


    # List of interger Tensors which define the indices of each value
    indices = randSparseIndices(shape, sparsity, attempts, device)

    nnz = indices.size()[1]

    # Each value in the SparseTensor
    values = torch.randn((nnz,), dtype=dtype, device=device)

    tensor = torch.sparse_coo_tensor(indices, values, dtype=dtype, device=device).coalesce()
    approx_sparsity = float(nz - tensor.values().size()[0]) / float(nz)
    print("Sparsity is approximate")
    print("Set sparisity: " + str(sparsity))
    print("Generated approximate sparsity: " + str(approx_sparsity))
    
    return tensor
