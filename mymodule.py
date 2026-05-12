""" Tran Le - 08/06/22 
This file contains all utility functions that will be called from other Jupyter notebooks.
Almost all of these functions are mainly learned from https://jovian.ai/aakashns/collections/deep-learning-with-pytorch
with some modification.
I also add some other functions to be easiser to used later on. 
"""

import os
import torch
import torchvision
# from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
import pandas as pd
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import glob
import cv2
import torchvision.transforms as tt
from torchvision.utils import make_grid
import torch.nn as nn
import torch.nn.functional as F
#----------------------------------------------------------
# GET THE DEVICE AND PUT MODEL/IMAGE IN THE DEVICE
# Function to get the device if available:
#----------------------------------------------------------
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
#   
def to_device(data, device):
    """Move data tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
# Create a DeviceDataLoader class to load the data
class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
#--------------------------------------------------------------
# IMAGE FOLDER EXPLORATION
#------------------------------------------------------------------
#-------------------------------------------------------------------------
# Get the number of images in each subfolder in the train and test set
#-------------------------------------------------------------------------
def num_files_f(data_dir='./Data/all'):
    """Find number of files in a data_dir, data direction
    the data_dir contain subfolder train and test (for train and test set)
    in each subfolder, there different classes
    These classes are the classes that we need to classify images into"""
    subfolders = os.listdir(data_dir)
    classes = os.listdir(data_dir + "/" + subfolders[1]) # get the class from the train folder 
    df = pd.DataFrame(columns=classes, index= subfolders)
    for s in subfolders: 
        for c in classes: 
            subfolder_file = os.listdir(data_dir + "/" + s + "/" + c)
            df.at[s, c] =  len(subfolder_file)
    return df

# Find min, max size of all images of the same kind in a folder 
def find_min_max_img_size(data_img_dir= "./data/train/VeryMildDemented/*.jpg"):
    """
    input:
    data_img_dir: the direction of the folder contains image and image kind, for ex, *.jpg
    data_img_dir ex: 
    """
    sizes = []  # list of the images
    for f in glob.iglob(data_img_dir):   #go to the directory and read all of file ending with .jpg
        img = cv2.imread(f) 
        sizes.append(img.shape) 
        # imgs.append(img)

    # sizes = []  #list of the images' sizes
    # for img in imgs:
    return min(sizes), max(sizes)    

def show_example(dataset, i):
    """
    dataset: a image data set gotten using ImageFolder
    i: the index of the image that we want to plot"""
    img, label = dataset[i]
    print('Label: ', dataset.classes[label], "("+str(label)+")")
    plt.imshow(img.permute(1, 2, 0) )


def show_batch(dl): 
    ""
    #dl: a data loader
    # Show image in a batch of dl, a dataloader..
    "" 
    for images, labels in dl:
        print(labels)
        fig, ax = plt.subplots(figsize=(20,9))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=8).permute(1, 2, 0))
        break

#--------------------------------------------------------------------------
#
#--------------------------------------------------------------------------

# Let's define the model by extending an ImageClassificationBase class which contains helper methods for training & validation.
# we will apply the forward in another class to extend this ImageClassificationBase class
class ImageClassificationBase(nn.Module):
    def training_step(self, batch): #self here is a model
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        # take the loss and accuracy from all of the different batches of the val data
        # and combine them by compute the mean
        # then return a single validation loss and validation accuracy
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
        
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

#--------------------------------------------------------------------------
# 
#-------------------------------------------------------------------------
import torch
from tqdm.notebook import tqdm

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        for batch in tqdm(train_loader):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#-------------------------------------------------------------------
# fit_one_cycle a function used in stead of fit function to improve the training step
# 
#------------------------------------------------------------------
def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []

    # Set up custom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                steps_per_epoch=len(train_loader))

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        lrs = []
        for batch in tqdm(train_loader):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()

        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history
# ---------------------------------------------------------------------
# Plot accuracy history on the val set
def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');
#--------------------------------------------------------------------
# plot the loss by epoch on the train and test set during training step
def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs');

# A function that returns the predicted label for a dataset (test set).

def predict_image(img, model, dataset, device):
    # dataset: a data set gotten by torchvision.datasets.DatasetFolder/ImageFolder
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    return dataset.classes[preds[0].item()]

def predict_image_label(img, model, Labels, device):
    # Labels: a vector contain Labels of images, for ex: Label= ['class1', 'class2', 'class3'] associative with 0, 1, 2
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    return Labels[preds[0].item()]


