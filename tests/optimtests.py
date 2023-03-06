import torch
from torch import nn
import time

def csr_optim_test():
    print(torch.version.__version__)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    assert device == "cuda"


    size = 1000000
    sparsity = 1.0-0.99

    a1 = [i for i in range(int(size*sparsity))]
    a1.append(size-1)
    a2 = [i for i in range(int(size*sparsity))]
    a2.append(size-1)

    indices = torch.tensor([
        a1,
        a2,
    ])

    values = torch.tensor([.5 for i in range(1+int(size*sparsity))])


    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.w = nn.Parameter(torch.sparse_coo_tensor(indices, values, dtype=torch.float,device='cuda').coalesce().to(device="cuda").to_sparse_csr(), requires_grad=True)
            print(self.w)

        # def loss(self):
        #     return torch.sqrt(self.w).values()[0]

        def forward(self, x):
            # print(x.size())
            # print(self.w.size())
            #x = x.transpose(0,1).to(device='cuda')
            #return torch.mv(self.w, x)
            term = torch.zeros((1,x.size()[0])).transpose(0, 1).to(device="cuda")
            vec = x.unsqueeze(1)
            out = torch.sparse.addmm(term, self.w, vec)
            return out.transpose(0, 1)


    model = Model().to(device=device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    loss_fn = nn.CrossEntropyLoss().to(device=device)

    x = torch.ones(size) - .1
    x = x.to('cuda') * 0.23
    y = (x * .5).to(device='cuda')
    y = y.unsqueeze(0)


    out = model(x)
    print(out)
    loss = loss_fn(out, y)

    print(loss)

    loss.backward()

    optimizer.step()

def csr_add_test():
    size = 1000000
    sparsity = 0.99

    a1 = [i for i in range(int(size*sparsity))]
    a1.append(size-1)
    a2 = [i for i in range(int(size*sparsity))]
    a2.append(size-1)


    indices1 = torch.tensor([
        a1,
        a2,
    ])

    values = torch.tensor([.5 for i in range(1+int(size*sparsity))])

    x = torch.sparse_coo_tensor(indices1, values, dtype=torch.float,device='cuda').coalesce().to(device="cuda").to_sparse_csr()

    indices2 = torch.tensor([
        a1,
        a2,
    ])

    values2 = torch.tensor([.5 for i in range(1+int(size*sparsity))])

    y =  torch.sparse_coo_tensor(indices2, values2, dtype=torch.float,device='cuda').coalesce().to(device="cuda").to_sparse_csr()
    

    for i in range(10000):
        print("#######################################TEST " + str(i))
        x = y.add_(x)
        print(x)
        time.sleep(0.05)

def csr_add_test2():
    nz = 10000
    sparsity = 0.99
    density = 1 - sparsity

    indices = torch.randint(0, nz, (2, int(nz*density)))
    values = torch.randn((int(nz*density)))

    x = torch.sparse_coo_tensor(indices, values, dtype=torch.float,device='cuda').coalesce().to_sparse_csr()

    for i in range(100):
        x = x.add_(x)
