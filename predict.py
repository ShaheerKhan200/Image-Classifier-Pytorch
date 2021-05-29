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
from PIL import Image
import argparse


parser = argparse.ArgumentParser(allow_abbrev=False, description='Predict the classes/labels for the picture')

# Must have arguments
parser.add_argument('Path',
                    metavar='path',
                    action="store", 
                    #default='flowers/valid/100/image_07905.jpg', 
                    type= str, 
                    #required= True,
                    help= 'Path to a single image for example "flowers/valid/100/image_07905.jpg"')

parser.add_argument('Checkpoint',
                    metavar='checkpoint', 
                    action="store", 
                    #default='checkpoint_desnet.pth', 
                    type=str,
                    #required= True,
                    help = 'Name of the Model saved ending with ".pth" for example "checkpoint_vgg.pth"')

# Optional arguments
parser.add_argument('-top_k','--top_k', 
                    action="store", 
                    default=5, 
                    type=int, 
                    metavar='', 
                    help = 'Return top K most likely classes')

parser.add_argument('-category_names','--category_names', 
                    action="store_true", 
                    help = 'Enter "-category_names" if you want to display a mapping of categories to real names')
parser.add_argument('-gpu','--gpu', 
                    action="store_true", 
                    help = 'Enter "gpu" if you want to use GPU for inference')

args = parser.parse_args()

# Variable assignment
image_path = args.Path

# If the user forgets to enter the checkpoint without .pth at the end
if(args.Checkpoint[-4:] != '.pth'):
    filepath = args.Checkpoint + '.pth'
else:
    filepath = args.Checkpoint

#if args.Checkpoint:
#    filepath = args.Checkpoint
#else:
#    print(
#try:
#  filepath = args.Checkpoint
#except IOError:
#  print("File doesn't exist")
    
top_k = args.top_k

if args.gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    
    if(checkpoint['model'] == 'desnet121'):
        #Load a pre-trained network
        model = models.densenet121(pretrained=True) 
        hidden_units = checkpoint['hidden_units']
        
        for param in model.parameters():
            param.requires_grad = False
        
        model.class_to_idx = checkpoint['class_to_idx']
        
        classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(1024, hidden_units, bias=True)),
                                ('relu1', nn.ReLU()),
                                ('dropout1', nn.Dropout(p=0.5)),
                                ('fc2', nn.Linear(hidden_units, 102, bias=True)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))

        model.classifier = classifier
        
        model.load_state_dict(checkpoint['state_dict'])
    
    elif(checkpoint['model'] == 'vgg19'):
        #Load a pre-trained network
        model = models.vgg19(pretrained=True) 
        hidden_units = checkpoint['hidden_units']
        
        for param in model.parameters():
            param.requires_grad = False
        
        model.class_to_idx = checkpoint['class_to_idx']
        
        classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(25088, hidden_units, bias=True)),
                                ('relu1', nn.ReLU()),
                                ('dropout1', nn.Dropout(p=0.5)),
                                ('fc2', nn.Linear(hidden_units, 102, bias=True)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))

        model.classifier = classifier
        
        model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image)
    
    transformations = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    
    return transformations(image)

def predict(device, image_path, model, top_k=top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)
    
    # Implement the code to predict the class from an image file 
    image = process_image(image_path).float().to(device)
    image = image.unsqueeze(0)
    model.eval()

    with torch.no_grad():
        output = model.forward(image)
    
    # Calculate the class probabilities for image
    ps = torch.exp(output)
    
    # Get the top predicted class, and the output percentage
    probs, classes = ps.topk(top_k, dim=1)
   
    probs = probs.numpy().tolist()
    classes = classes.numpy().tolist()
    
    return probs[0], classes[0]

with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

def predict_modified(device, image_path, model, cat_to_name, top_k=top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)
    
    # Implement the code to predict the class from an image file 
    image = process_image(image_path).float().to(device)
    image = image.unsqueeze(0)
    model.eval()

    with torch.no_grad():
        output = model.forward(image)
    
    # Calculate the class probabilities for image
    ps = torch.exp(output)
    
    # Get the top predicted class, and the output percentage
    probs, classes = ps.topk(top_k, dim=1)
    probs = probs.tolist()
    classes = classes.tolist()
    
    # DataFrame Processing
    
    ## Set up cat_to_name dataframe
    cat_to_name_df = pd.DataFrame.from_dict((cat_to_name),orient='index', columns=['Label Name'])
    cat_to_name_df['labels'] = cat_to_name_df.index
    cat_to_name_df[['labels']] = cat_to_name_df[['labels']].apply(pd.to_numeric)
    
    
    df_probabilities = pd.DataFrame(probs[0],classes[0], columns =['Probabilities'])
    df_probabilities['labels'] = df_probabilities.index
    
    merged_df = pd.merge(cat_to_name_df, df_probabilities, how = 'right')
    merged_df = merged_df.sort_values('Probabilities', ascending = False)

    if args.category_names:
        merged_df = merged_df.reset_index(drop=True) #df with labels and names
    else:
        merged_df = merged_df.reset_index(drop=True)
        merged_df = merged_df.drop('Label Name', axis=1)

    return merged_df


#To print dataframe
model = load_checkpoint(filepath)
df = predict_modified(device, image_path, model, cat_to_name, top_k)
print(df)

# to insert in command line for testing
# for command line 
# python predict.py flowers/valid/100/image_07905.jpg vgg_final
# python predict.py flowers/valid/100/image_07905.jpg desnet121_final