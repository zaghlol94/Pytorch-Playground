import numpy as np
import torch
import helper
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as F


def activation(x):
    return 1 / (1 + torch.exp(-x))


def softmax(x):
    return torch.exp(x) / torch.sum(torch.exp(x), dim=1).view(-1, 1)


torch.manual_seed(7)

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=False, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)



# # How calculate loss in pytorch
#
# model = nn.Sequential(nn.Linear(784, 128),
#                       nn.ReLU(),
#                       nn.Linear(128, 64),
#                       nn.ReLU(),
#                       nn.Linear(64, 10))
#
# # Define the loss
# criterion = nn.CrossEntropyLoss()
#
# # Get our data
# images, labels = next(iter(trainloader))
# # Flatten images
# images = images.view(images.shape[0], -1)
#
# # Forward pass, get our logits
# logits = model(images)
# # Calculate the loss with the logits and the labels
# loss = criterion(logits, labels)
#
# print(loss)


# Build a feed-forward network
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

# Define the loss
criterion = nn.NLLLoss()

# Get our data
images, labels = next(iter(trainloader))
# Flatten images
images = images.view(images.shape[0], -1)

# Forward pass, get our log-probabilities
logps = model(images)
# Calculate the loss with the logps and the labels
loss = criterion(logps, labels)

print(loss)


# Autograd
x = torch.randn(2,2, requires_grad=True)
print(x)
y = x**2
print(y)
print(y.grad_fn)

z = y.mean()
print(z)

print(x.grad)
z.backward()
print(x.grad)
print(x/2)

# Optimizer
from torch import optim

# Optimizers require the parameters to optimize and a learning rate
optimizer = optim.SGD(model.parameters(), lr=0.01)

print('Initial weights - ', model[0].weight)

images, labels = next(iter(trainloader))
images.resize_(64, 784)

# Clear the gradients, do this because gradients are accumulated
optimizer.zero_grad()

# Forward pass, then backward pass, then update weights
output = model(images)
loss = criterion(output, labels)
loss.backward()
print('Gradient -', model[0].weight.grad)
# Take an update step and few the new weights
optimizer.step()
print('Updated weights - ', model[0].weight)
print(model)
