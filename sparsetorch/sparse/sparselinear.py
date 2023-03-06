import torch
from torch import nn
from torch import Tensor
from sparsetorch.sparse.sparseutil import randCOOTensor
from sparsetorch.sparse import sparseops
import warnings

class SparseLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, grad_enabled: bool, bias: bool=True, sparsity=0.9, sparsity_approximation_attempts = 1, sparse_layout = torch.sparse_csr, device=None, dtype=None) -> None:
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super(SparseLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity
        self.sparse_layout = sparse_layout
        self.grad_enabled = grad_enabled
        self.sparsity_approximation_attempts = sparsity_approximation_attempts

        torchver = int(torch.version.__version__[0])
        if torchver < 2:
            if grad_enabled == True and sparse_layout == torch.sparse_csr:
                pass
                #raise Exception("operations with torch.sparse_csr layout do not support gradients currently, torch.sparse_coo layout do support gradients")
            if grad_enabled == False and sparse_layout == torch.sparse_coo:
                warnings.warn("For applications with no need for gradients torch.sparse_csr will work perform much faster than torch.sparse_coo")
            
        weight = randCOOTensor((self.out_features, self.in_features), sparsity=self.sparsity, attempts=self.sparsity_approximation_attempts, **self.factory_kwargs)

        if self.sparse_layout == torch.sparse_csr:
            weight = weight.to_sparse_csr()
        elif self.sparse_layout != torch.sparse_coo:
            raise NotImplementedError("Only sparse_layout COO and CSR are implemented in SparseLinear")

        self.weight = nn.Parameter(weight, requires_grad=self.grad_enabled)

        self.bias = torch.zeros(self.out_features, **self.factory_kwargs).to(self.factory_kwargs['device'])
        if bias:
            self.bias = nn.Parameter(torch.randn(self.out_features, **self.factory_kwargs), requires_grad=self.grad_enabled)


    # def reset_parameters(self) -> None:
    #     # Reinitializing weight and bias matricies
    #     weight = randCOOTensor((self.out_features, self.in_features), sparsity=self.sparsity, attempts=self.sparsity_approximation_attempts, **self.factory_kwargs)
    #     if self.sparse_layout == torch.sparse_csr:
    #         weight = weight.to_sparse_csr()
    #     elif self.sparse_layout != torch.sparse_coo:
    #         raise NotImplementedError("Only sparse_layout COO and CSR implemented")
    #     self.weight = nn.Parameter(weight, requires_grad=self.grad_enabled)        
        
    #     self.bias = None
    #     if bias:
    #         self.bias = nn.Parameter(torch.rand(self.out_features, **self.factory_kwargs), requires_grad=self.grad_enabled)


    @property
    def getParameters(self):
        return self.weight, self.bias


    def forward(self, x: Tensor) -> Tensor:        
        # Use `nn.functional.linear(x, self.weight, bias=self.bias)` once sparse grads are fully supported, most probably with pytorch 2
        return sparseops.batch_addmm(self.weight, x, self.bias)