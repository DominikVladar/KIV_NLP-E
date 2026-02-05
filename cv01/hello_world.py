import torch
#
# a = torch.tensor(5)
# b = torch.tensor(4)

dev = "cpu"
a = torch.tensor(5).to(dev)
b = torch.tensor(4).to(dev)

c = a+b

print(c)
