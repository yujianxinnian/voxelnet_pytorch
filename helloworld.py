import torch

a = torch.randn(1,2,3)
print('a.shape',a.shape)
print('a=',a)
b = a.permute(0,2,1)
print('b.shape',b.shape)
print('b=',b)