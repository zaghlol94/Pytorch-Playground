import torch

b = torch.arange(2 * 2 * 3).view(2, 2, 3)
print(b)

print(torch.sum(b, (0, 1)))

print(torch.sum(b, (2, 1)))


c = torch.arange(2 * 2 * 3).view(2,6)


print(c)
print()
print(torch.sum(c,1))
print(torch.sum(c,1).shape)


print(torch.sum(c,1))

d = torch.randn((2,6))
print(d)
print(torch.sum(torch.exp(d),dim=1).view(-1,1))
print(torch.exp(d)/torch.sum(torch.exp(d),dim=1))
