

import argparse

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

import json

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'




with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


def arg_parser():
    parser = argparse.ArgumentParser(description="predict.py")
    parser.add_argument('image', type=str)
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--top_k', dest="top_k", type=int,default=3)
    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
    parser.add_argument('--gpu', default="cuda", action="store", dest="gpu")

    
    return parser.parse_args()

args = arg_parser()

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
    epochs = checkpoint['epochs']
     
    #Freezing model parameters
    for param in model.parameters():
        param.requires_grad = False
        
    return model, checkpoint['class_to_idx']

model, class_to_idx = load_checkpoint(args.checkpoint)
class_to_idx

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    
    
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                          [0.229, 0.224, 0.225])])
    
    img = transform(image)
  
    return img
    
                                      
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def predict(image_path, model, topk=args.top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    model.to(args.gpu)
    model.eval()
    
    with torch.no_grad():
        # Process image
        img = process_image(image_path)
        image_tensor = img.to(args.gpu)
        
        # Add batch of size 1 to image
        model_input = image_tensor.unsqueeze(0)
        output = model.forward(model_input)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk, dim=1)
        # Convert indices to classes
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        top_labels=[]
        
    return top_p, top_class

path = args.image


probs, classes =  predict(Image.open(path), model)
predicted_flower = classes[0].cpu().numpy()

predicted_flower1 = map(str, predicted_flower)


name_idx = {name: idx for idx, name in class_to_idx.items()}


top_flowers = [name_idx[index] for index in predicted_flower]



top_flowers = [cat_to_name[index] for index in top_flowers]

probs = probs[0].cpu().numpy()

print("\n\n")
for x in range(args.top_k):
    print(top_flowers[x], probs[x])




