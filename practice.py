import torch 

a = torch.randn(4,1,2,2)
b= torch.randint(0,1,(4,1))
embed = torch.nn.Embedding(2,4)
c = embed(b)
print(c)
d= c.view(4,1,2,2)

print("shape of a",a.shape)

print("shape of b",b.shape)
print("shape of c",c.shape)
print("shape of d",d.shape)
