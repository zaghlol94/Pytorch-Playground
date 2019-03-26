from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torch.utils.data import dataloader
from torch.nn import Linear
from glob import glob
import torch.nn as nn
from torch import optim
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.models import alexnet, resnet50, resnet18, inception, squeezenet
import time

plt.ion()
#
path = 'test/'
files = glob(os.path.join(path, '*/*/*.png'))
print(f'Total number of images is {len(files)}')
no_of_images = len(files)
shuffle = np.random.permutation(no_of_images)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(60),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(299),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(60),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
print("Initializing Datasets and Dataloaders...")

image_datasets = {x: ImageFolder(os.path.join(path, x),
                                 data_transforms[x])
                  for x in ['train', 'valid']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes

# model initializations
print('# Model Initialization')
model_ft = inception.inception_v3(pretrained=True)
model_ft.aux_logits = False
num_classes = 2
for param in model_ft.parameters(): param.requires_grad = False
model_ft.fc = nn.Linear(2048, num_classes)
criterion = nn.CrossEntropyLoss()
# optimizer
params_to_update = model_ft.parameters()

print()
print("Params to learn:")
for name, param in model_ft.named_parameters():
    if param.requires_grad == True:
        print(name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, 'min', 0.1, patience=4, verbose=True,
                                                        threshold=1e-2)

train_loss = []
valid_loss = []


def train_model(model, criterion, optimizer, scheduler, num_epochs=50):
    model.to(device)
    since = time.time()
    best_model_wts = model.state_dict()
    best_loss = 999
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0.0

            for data in dataloaders[phase]:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                inputs, labels = Variable(inputs), Variable(labels)
                optimizer.zero_grad()
                # forward path
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs.data, 1)

                # backward path
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # copy the best model
            if phase == 'valid' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()
                torch.save(best_model_wts, os.path.join(path,
                                                        'output2/valid_acc_{:.4f}.hdf5'.format(
                                                            epoch_acc)))
                print('loss changed to {:.4f} ....saving weights'.format(epoch_loss))
            if phase == 'valid':
                scheduler.step(epoch_loss)
                valid_loss.append(epoch_loss)
            if phase == 'train':
                train_loss.append(epoch_loss)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # save best model
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, valid_loss, train_loss


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(2)


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['valid']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


model_ft, valid_loss, train_loss = train_model(model_ft, criterion, optimizer_ft, scheduler=exp_lr_scheduler)
visualize_model(model_ft)
plt.plot(train_loss, label='Training loss')
plt.plot(valid_loss, label='Validation loss')
plt.legend(frameon=False)
plt.show()
plt.imsave('test/valid_loss_featureex.png', valid_loss)
plt.imsave('test/train_loss_featureex.png', train_loss)