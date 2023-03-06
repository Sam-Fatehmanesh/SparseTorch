import torch
from torch import nn
from sparsetorch.util import *
from sparsetorch.sparse.sparselinear import SparseLinear
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch import functional


def testBPTrainMNIST(sparse):
    torch.set_default_dtype(torch.float)
    dtype = torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = FashionMNIST("testdata", download=True, transform=transforms.ToTensor())
    train_data, val_data = random_split(dataset, [59000, 1000])


    loaders = {
        'train' : torch.utils.data.DataLoader(train_data, 
                                            batch_size=256, 
                                            shuffle=True, 
                                            num_workers=1),
        
        'test'  : torch.utils.data.DataLoader(val_data, 
                                            batch_size=256, 
                                            shuffle=True, 
                                            num_workers=1),
    }

    class sparseNN(nn.Module):
        def __init__(self):
            super(sparseNN, self).__init__()

            sparse_layout = torch.sparse_csr
            sparsity = 0.99
            self.layer = nn.Sequential(
                nn.BatchNorm2d(1),
                nn.Flatten(),#100000
                SparseLinear(28*28, 200000, grad_enabled=True, sparse_layout=sparse_layout, sparsity=sparsity, device=device),
                #nn.Linear(28*28, 1000, bias=False),

                nn.Sigmoid(),
                SparseLinear(200000, 10, grad_enabled=True, sparse_layout=sparse_layout, sparsity=sparsity,device=device),
                #nn.Linear(1000, 10),

                nn.Softmax(dim=1),
            ) 

        def forward(self, x):
            return self.layer(x)
    
    class dNN(nn.Module):
        def __init__(self):
            super(dNN, self).__init__()

            sparse_layout = torch.sparse_csr
            sparsity = 0.0
            self.layer = nn.Sequential(
                nn.BatchNorm2d(1),
                nn.Flatten(),#100000
                #SparseLinear(28*28, 1000, grad_enabled=True, sparse_layout=sparse_layout, sparsity=sparsity, device=device),
                nn.Linear(28*28, 600),

                nn.Sigmoid(),
                #SparseLinear(1000, 10, grad_enabled=True, sparse_layout=sparse_layout, sparsity=sparsity,device=device),
                nn.Linear(600, 10),

                nn.Softmax(dim=1),
            ) 

        def forward(self, x):
            return self.layer(x)

    model = None
    if sparse:
        model = sparseNN().to(device=device)
    else:
        model = dNN().to(device=device)
    

    loss_func = nn.CrossEntropyLoss().to(device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr = .01)

    epochs = 1


    model.train()

    for epoch in range(epochs):
        for i, (images, lables) in enumerate(loaders['train']):
            #with torch.autograd.detect_anomaly():
            data = images.to(device).float()
            targets = lables.to(device)
            optimizer.zero_grad()
            #print(data)
            output = model(data)
            
            loss = loss_func(output, targets)
            
            # backprob
            loss.backward()
            # update weights
            optimizer.step()

            #Grads for self.weight of SparseLinear are being generated but are not being added to the matrix
            # if list(model.parameters())[1].grad is not None:
            #     print("grads exist")
            #     print(list(model.parameters())[1].grad)
            #     print(list(model.parameters())[1])

            if i % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, i * len(images), len(loaders['train'].dataset),
                    100. * i / len(loaders['train']), loss.item()))

    model.eval()
    size = len(loaders['test'].dataset)
    num_batches = len(loaders['test'])
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in loaders['test']:
            y=y.to(device)
            pred = model(X.to(device=device).float())
            test_loss += loss_func(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct
