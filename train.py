


import tqdm
from tqdm import tqdm

import PIL
from PIL import Image

import numpy as np

import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from torch.autograd import Variable


import argparse
import json


def arg_parser():
    parser = argparse.ArgumentParser(description="Train.py")
    parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
    parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=512)
    parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=5)
    parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")

    return parser.parse_args()


args = arg_parser()

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(90),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], 
                                                          [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(224),
                                     transforms.CenterCrop(210),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])


valid_transforms = transforms.Compose([transforms.Resize(224),
                                     transforms.CenterCrop(210),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])


# TODO: Load the datasets with ImageFolder
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
test_datasets = datasets.ImageFolder(test_dir, transform=valid_transforms)
valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_datasets, batch_size=64)
validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=64)


def primaryloader_model(architecture="vgg16"):
    
    
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
        
        for param in model.parameters():
            param.requires_grad = False 
        return model


if architecture=args.arch == 'vgg16':
    model = primaryloader_model(architecture=args.arch)
else: model = primaryloader_model(architecture='resnet50')


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
# Use GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=True)

print(model)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
    
model.fc = nn.Sequential(nn.Linear(2048, args.hidden_units),
                      nn.ReLU(),
                      nn.Linear(args.hidden_units, args.hidden_units),
                      nn.ReLU(),
                      nn.Linear(args.hidden_units, 102),
                      nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)

model.to(device);

# TODO: Do validation on the test set
epochs = args.epochs
steps = 0
running_loss = 0
print_every = 5
#for i in keep_awake(range(2)):
for epoch in range(epochs):
    for inputs, labels in tqdm(trainloader):
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)

                    test_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(validloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()
            
def networkTest(model, testloader, criterion):
    test_loss = 0
    accuracy = 0

    model.eval()

    with torch.no_grad():
        for images, labels in testloader:
            model.to(device)
            images, labels = images.to(device), labels.to(device)

            output = model.forward(images)
            loss = criterion(output, labels)
            test_loss += loss.item()

            # Calculate accuracy
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    model.train()
    return accuracy/len(testloader)

networkTest(model,testloader, criterion)
            
            
def save_checkpoint(input_num, output, model, epochs, image_datasets):
    model.class_to_idx = image_datasets.class_to_idx
    
    checkpoint = {'input_size': input_num,
                  'output_size': output,
                  'model' : model,
                  'epochs' : epochs,
                  'classifier' : model.fc,
                  'state_dict': model.state_dict(),
                  'optimizer' : optimizer.state_dict,
                  'class_to_idx' : model.class_to_idx}
    return checkpoint
    
    
torch.save(save_checkpoint(2048, 102, model, epochs, train_datasets), 'checkpoint.pth')
