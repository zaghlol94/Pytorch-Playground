import numpy as np
import torch
import helper
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as F

def activation(x):
    return 1/(1+torch.exp(-x))


def softmax(x):
    return torch.exp(x)/torch.sum(torch.exp(x), dim=1).view(-1, 1)


# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])
# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=False, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)


dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)

plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')
plt.show()


# Flatten the input images
print(images.shape)
inputs = images.view(images.shape[0], -1)
print(inputs.shape)
# Create parameters
w1 = torch.randn(784, 256)
b1 = torch.randn(256)

w2 = torch.randn(256, 10)
b2 = torch.randn(10)

h = activation(torch.mm(inputs, w1) + b1)

out = torch.mm(h, w2) + b2
print(out.shape)

probabilities = softmax(out)

# Does it have the right shape? Should be (64, 10)
print(probabilities.shape)
# Does it sum to 1?
print(probabilities.sum(dim=1))

'''
Use nn.Module to build FNN 
'''


# class Network(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         # Inputs to hidden layer linear transformation
#         self.hidden = nn.Linear(784, 256)
#         # Output layer, 10 units - one for each digit
#         self.output = nn.Linear(256, 10)
#
#         # Define sigmoid activation and softmax output
#         self.sigmoid = nn.Sigmoid()
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         # Pass the input tensor through each of our operations
#         x = self.hidden(x)
#         x = self.sigmoid(x)
#         x = self.output(x)
#         x = self.softmax(x)
#
#         return x
#
#
# model = Network()
# print(model)
#
# # another method of writing
# import torch.nn.functional as F
#
#
# class Network2(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Inputs to hidden layer linear transformation
#         self.hidden = nn.Linear(784, 256)
#         # Output layer, 10 units - one for each digit
#         self.output = nn.Linear(256, 10)
#
#     def forward(self, x):
#         # Hidden layer with sigmoid activation
#         x = F.sigmoid(self.hidden(x))
#         # Output layer with softmax activation
#         x = F.softmax(self.output(x), dim=1)
#
#         return x
#
#

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Defining the layers, 128, 64, 10 units each
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        # Output layer, 10 units - one for each digit
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)

        return x


model = Network()

print(model.fc1.weight)
print(model.fc1.bias)
model.fc1.bias.data.fill_(0)
print(model.fc1.bias)

# Grab some data
dataiter = iter(trainloader)
images, labels = dataiter.next()

# Resize images into a 1D vector, new shape is (batch size, color channels, image pixels)
images.resize_(64, 1, 784)
# or images.resize_(images.shape[0], 1, 784) to automatically get batch size

# Forward pass through the network
img_idx = 0
ps = model.forward(images[img_idx,:])

img = images[img_idx]
helper.view_classify(img.view(1, 28, 28), ps)

