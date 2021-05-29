# Importing Libraries
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict

import numpy as np
import pandas as pd
import json

import argparse

parser = argparse.ArgumentParser(allow_abbrev=False, 
                                description='Prints out training loss, validation loss, and validation accuracy as the network trains')

#must have arguments
parser.add_argument('Path',
                    metavar='path',
                    action="store", 
                    default='./ImageClassifier/flowers', 
                    type= str, 
                    #required= True,
                    help= 'Path to data folder for example "/flowers"')

#optional arguments
parser.add_argument('-save_dir','--save_dir', 
                    action="store",
                    default='./checkpoint_desnet.pth', 
                    type=str,
                    help = 'Enter the name of checkpoint to save the model under')

parser.add_argument('-arch','--arch', 
                    action="store", 
                    default= 'desnet121',
                    choices=['vgg19', 'desnet121'],
                    type=str,
                    help = 'Enter architecture of choice to train the model')

parser.add_argument('-learning_rate','--learning_rate', 
                    action="store", 
                    default=0.002, 
                    type=float, 
                    metavar='', 
                    help = 'Train the model on the specified learning rate')

parser.add_argument('-hidden_units','--hidden_units', 
                    action="store", 
                    default=1000, 
                    type=int, 
                    metavar='', 
                    help = 'Train the model on the specified number of hidden_units. Use "4096" for vgg19 and Use "1000" for desnet121')

parser.add_argument('-epochs','--epochs', 
                    action="store", 
                    default=8, 
                    type=int, 
                    metavar='', 
                    help = 'Train the model on the specified number of epochs')

parser.add_argument('-gpu','--gpu', 
                    action="store_true", 
                    help = 'Enter "gpu" if you want to use GPU for training')

args = parser.parse_args()

if args.Path:
    data_dir = args.Path
else:
    data_dir = '/ImageClassifier/flowers'


if(args.save_dir[-4:] != '.pth'):
    save_dir = args.save_dir + '.pth'
else:
    save_dir = args.save_dir
    
arch = args.arch
learning_rate = args.learning_rate

hidden_units = args.hidden_units
epochs = args.epochs

if args.gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def processing_load_data(data_dir):
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define your transforms for the training, validation, and testing sets
    training_data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    testing_data_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    validation_data_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_image_datasets = datasets.ImageFolder(train_dir, transform=training_data_transforms)
    test_image_datasets = datasets.ImageFolder(test_dir, transform=testing_data_transforms)
    valid_image_datasets = datasets.ImageFolder(valid_dir, transform=validation_data_transforms)


    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_image_datasets, batch_size=64, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_image_datasets, batch_size=64, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_image_datasets, batch_size=64, shuffle = True)

    #label mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    return trainloader, testloader, validloader, train_image_datasets, cat_to_name

#Testing the Function
trainloader, testloader, validloader, train_image_datasets, cat_to_name = processing_load_data(data_dir)

def build_model(arch, hidden_units, device, learning_rate):
    
    if(arch == 'desnet121'):
        #Load a pre-trained network
        model = models.densenet121(pretrained=True)

        # Define a new, untrained feed-forward network as a classifier, 
        # using ReLU activations and dropout
        for param in model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(1024, hidden_units, bias=True)),
                                ('relu1', nn.ReLU()),
                                ('dropout1', nn.Dropout(p=0.5)),
                                ('fc2', nn.Linear(hidden_units, 102, bias=True)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))

        model.classifier = classifier

    elif (arch == 'vgg19'):
        #Load a pre-trained network
        model = models.vgg19(pretrained=True)

        # Define a new, untrained feed-forward network as a classifier, 
        # using ReLU activations and dropout
        for param in model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(25088, hidden_units, bias=True)),
                                ('relu1', nn.ReLU()),
                                ('dropout1', nn.Dropout(p=0.5)),
                                ('fc2', nn.Linear(hidden_units, 102, bias=True)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))

        model.classifier = classifier
        
    # Use GPU if it's available
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model.to(device)
    return model, criterion, optimizer, device

#Implimentation of the function
model, criterion, optimizer, device = build_model(arch, hidden_units, device, learning_rate)

def training(model, criterion, optimizer, trainloader, testloader, validloader, device, epochs):
    
    # Train the classifier layers using backpropagation using
    # the pre-trained network to get the features 
    epochs = epochs
    steps = 0
    running_loss = 0
    print_every = 50

    for epoch in range(epochs):
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for jj, (inputs2, labels2) in enumerate(validloader):
                        inputs2, labels2 = inputs2.to(device), labels2.to(device)
                        
                        outputs2 = model.forward(inputs2)
                        batch_loss = criterion(outputs2, labels2)
                        
                        test_loss += batch_loss.item()
                        
                        # Calculate accuracy
                        ps = torch.exp(outputs2)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels2.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Validation loss: {test_loss/len(validloader):.3f}.. "
                    f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    return model

#testing
if (device == 'cuda'):
    print("Training on GPU")
elif (device =='cpu'):
    print("Training on CPU")
    
model = training(model, criterion, optimizer, trainloader, testloader, validloader, device, epochs)

print("-"*70)
if (arch == 'desnet121'):
    print("Architecture Used is denset121")
elif (arch == 'vgg19'):
    print("Architecture Used is vgg19")
print("\n")

print(f"Model Saved is called = {save_dir}\
        Learning Rate = {learning_rate}\
        Hidden Units = {hidden_units}\
        Number of epochs = {epochs}")
    
def save_checkpoints(model, arch, train_image_datasets, save_dir):
    model.class_to_idx = train_image_datasets.class_to_idx

    checkpoint = {'model' : arch,
                  'hidden_units' : hidden_units,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, save_dir)

# save checkpoint
save_checkpoints(model, arch, train_image_datasets, save_dir)
print("-"*70)
print("\n")
print("Model Successfully Saved!")


# to insert in command line for testing
## python train.py /flowers -arch desnet121 -epochs 1 -gpu
## python train.py /flowers -arch desnet121 -epochs 1 
## python train.py /flowers -save_dir vgg_final -arch vgg19 -hidden_units 4096 -epochs 8 -gpu