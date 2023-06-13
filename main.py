import torch
import torchvision
import torchvision.transforms as transforms



import os
import argparse
import matplotlib.pyplot as plt

from model import MLP
from model import prepare_model

from train import train_epoch
from train import test_epoch

def plot(data):
    fig, axes = plt.subplots(5, 5, figsize=(5, 5))

    # Plot the images in the grid
    for i in range(5):
        for j in range(5):
            image, label = data[i*5+j]
            axes[i, j].imshow(image.squeeze(), cmap='gray')
            axes[i, j].set_title(f"Label: {label}")
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()

def get_data():
    transform = transforms.ToTensor()
    download = True
    if os.path.exists('./data'):
        download = False
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=download)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=download)
    return train_dataset, test_dataset

def data_to_dataloader(data, shuffle, args):
    dataloader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=shuffle)
    return dataloader
    
def main(args):
    train, test = get_data()
    train_dataloader = data_to_dataloader(data=train, shuffle=True, args=args)
    
    test_dataloader = data_to_dataloader(data=test, shuffle=False, args=args)
    
    model = MLP(input_size=784, hidden_size=150, output_size=10)
    loss_func, optim = prepare_model(model=model)
        
    max_epoch = 5
    
    for epoch in range(max_epoch):
        print(f"running {epoch} :")
        train_epoch(model=model, train_dataloader=train_dataloader, loss_func=loss_func, optimizer=optim, epoch=epoch)
        test_epoch(model=model, test_dataloader=test_dataloader, loss_func=loss_func, optimizer=optim, epoch=epoch)
        

if __name__ == '__main__':
    if os.path.exists('./data'):
        print("exits")
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--prompts', type=str, default='kudod', help='Input file path')
    parser.add_argument('--batch_size', type=int, default='32', help='Input file path')
    args = parser.parse_args()
    main(args)
    

    