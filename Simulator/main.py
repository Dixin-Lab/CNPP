import torch

a=torch.tensor([
    [1,2,3],
    [1,2,3],
    [1,2,6]
])
a[range(3),range(3)]=0
print(a[range(3),range(3)])
print(a)