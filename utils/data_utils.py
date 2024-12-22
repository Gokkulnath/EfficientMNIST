import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from config import Config

def get_data_loaders(batch_size, num_workers):
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load the full training dataset
    full_dataset = datasets.MNIST('./data', 
                                train=True, 
                                download=True,
                                transform=transform)
    
    # Split into train and validation sets
    val_size = 10000
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(Config.SEED)
    )
    
    # Load test dataset
    test_dataset = datasets.MNIST('./data', 
                                train=False,
                                transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader 