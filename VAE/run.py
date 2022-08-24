import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

from main import fit, val
from Config import epochs, batch_size, lr
from VAE import VAE

if __name__ == "__main__":

    train_loss = []
    val_loss = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # transforms
    transform = transforms.Compose([transforms.ToTensor()])

    # train and validation data
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    val_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform )

    # training and validation data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss(reduction='sum')

    for epoch in range(epochs):
        
        train_epoch_loss = fit(epoch, model, optimizer, criterion, train_loader, device)
        val_epoch_loss = val(epoch, model, optimizer, criterion, val_loader, device)

        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        
        print(f"Epoch {epoch+1} | Train Loss: {train_epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}")