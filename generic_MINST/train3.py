import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
from generic_MINST.Model import Net
import  torch.nn.functional as ff
# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# choose the training and test datasets
train_data = datasets.MNIST(root='data', train=True,
                                   download=False, transform=transform)
test_data = datasets.MNIST(root='data', train=False,
                                  download=False, transform=transform)

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
    num_workers=num_workers)

model = Net()
# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()

# specify optimizer (stochastic gradient descent) and learning rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

n_epochs = 50

model.train()  # prep model for training

valid_loss_min = np.Inf  # set initial "min" to infinity
for epoch in range(n_epochs):
    # monitor training loss
    train_loss = 0.0
    train_acc = 0
    ###################
    # train the model #
    ###################
    for data, target in train_loader:
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()
        _, pred = torch.max(output, 1)
        correct = pred == target.view(*pred.shape)
        train_acc += torch.mean(correct.type(torch.FloatTensor)).item()

    model.eval()
    test_loss = 0
    acc = 0
    with torch.no_grad():
        for data , target in test_loader:
            output = model(data)
            loss = criterion(output,target)
            test_loss += loss.item()
            _, pred = torch.max(output, 1)
            correct = pred == target.view(*pred.shape)
            acc += torch.mean(correct.type(torch.FloatTensor)).item()

        test_loss = test_loss/len(test_loader)
        acc = acc / len(test_loader)


    # print training statistics
    # calculate average loss over an epoch
    train_loss = train_loss / len(train_loader)
    train_acc = train_acc / len(train_loader)

    print('Epoch: {} \tTraining Loss: {:.6f} \tTesting Loss: {:.6f} \tTesting acc: {:.6f} \tTraining acc: {:.6f}'.format(
        epoch + 1,
        train_loss,
        test_loss,
        acc,
        train_acc
    ))
    if test_loss < valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            test_loss))
        torch.save(model.state_dict(), 'best_model.pt')
        valid_loss_min = test_loss
    model.train()
