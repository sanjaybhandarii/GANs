import torch 

a = torch.randn(4,1,1,1).repeat(1,2,2,2)
print(a.shape)
print(a.view(a.shape[0],-1))