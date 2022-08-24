import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from tqdm import tqdm

from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

from Config import batch_size, epochs
from VAE import VAE
from Loss import VAELoss


def fit(epoch, model, optim, criterion, dataloader, device):
    model.train()
    running_loss = 0.0
    
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train')

    for i, data in enumerate(dataloader):
        data, _ = data
        data = data.to(device)
        data = data.view(data.size(0), -1)

        optim.zero_grad()
        x_hat, mu, logvar = model(data)

        bce_loss = criterion(x_hat, data)
        loss = VAELoss(bce_loss, mu, logvar)
        running_loss += loss.item()

        loss.backward()
        optim.step()

        batch_bar.set_postfix(
            epoch=f"{epoch+1}/{epochs}",
            loss="{:.04f}".format(float(running_loss / (i + 1))),
            lr="{:.06f}".format(float(optim.param_groups[0]['lr']))
        )

        batch_bar.update()
    
    batch_bar.close()
    
    train_loss = running_loss/len(dataloader.dataset)
    
    return train_loss

def val(epoch, model, optim, criterion, dataloader, device):
    model.eval()
    running_loss = 0.0

    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Val')

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            data, _ = data
            data = data.to(device)
            data = data.view(data.size(0), -1)

            x_hat, mu, logvar = model(data)

            bce_loss = criterion(x_hat, data)
            loss = VAELoss(bce_loss, mu, logvar)
            running_loss += loss.item()
        
            # save the last batch input and output of every epoch
            # if i == int(len(val_data)/dataloader.batch_size) - 1:
            #     num_rows = 8
            #     both = torch.cat((data.view(batch_size, 1, 28, 28)[:8], x_hat.view(batch_size, 1, 28, 28)[:8]))
            #     save_image(both.cpu(), f"./output/output_{epoch+1}.png", nrow=num_rows)
            #     # plt.imshow
            
            batch_bar.set_postfix(
            epoch=f"{epoch+1}/{epochs}",
            loss="{:.04f}".format(float(running_loss / (i + 1))),
            lr="{:.06f}".format(float(optim.param_groups[0]['lr']))
        )

        batch_bar.update()
    
    batch_bar.close()

    val_loss = running_loss/len(dataloader.dataset)

    return val_loss