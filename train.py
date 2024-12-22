import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from datetime import datetime
from models.model import MNISTNet
from utils.data_utils import get_data_loaders
from utils.model_utils import print_model_summary
from config import Config

# Disable tqdm in GitHub Actions
GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS') == 'true'

def write_log(message, filename='training_logs.log'):
    """Write message to log file with timestamp"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(filename, 'a') as f:
        f.write(f'[{timestamp}] {message}\n')

def train_epoch(model, device, train_loader, optimizer, criterion):
    model.train()
    # Use tqdm only if not in GitHub Actions
    iterator = train_loader if GITHUB_ACTIONS else tqdm(train_loader)
    correct = 0
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(iterator):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_loss += loss.item()
        
        if not GITHUB_ACTIONS:
            iterator.set_description(f'Loss: {loss.item():.4f}')
        elif batch_idx % 50 == 0:  # Print progress occasionally in CI
            print(f'Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    # Calculate epoch metrics
    n_samples = len(train_loader.dataset)
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / n_samples
    
    return avg_loss, accuracy

def validate(model, device, loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loader)
    accuracy = 100. * correct / len(loader.dataset)
    
    return test_loss, accuracy

def main():
    # Set random seeds
    torch.manual_seed(Config.SEED)
    
    # Initialize model
    model = MNISTNet(dropout_rate=Config.DROPOUT_RATE).to(Config.DEVICE)
    
    # Print model summary with receptive field information
    print_model_summary(model, input_size=(1, 1, 28, 28))
    
    # Print training parameters
    print("\nTraining Parameters:")
    print("-"*80)
    print(f"{'Parameter':<30} {'Value':<20}")
    print("-"*80)
    print(f"{'Batch Size':<30} {Config.BATCH_SIZE:<20}")
    print(f"{'Learning Rate':<30} {Config.LEARNING_RATE:<20}")
    print(f"{'Momentum':<30} {Config.MOMENTUM:<20}")
    print(f"{'Weight Decay':<30} {Config.WEIGHT_DECAY:<20}")
    print(f"{'Dropout Rate':<30} {Config.DROPOUT_RATE:<20}")
    print(f"{'Number of Epochs':<30} {Config.EPOCHS:<20}")
    print(f"{'Device':<30} {Config.DEVICE:<20}")
    print(f"{'Number of Workers':<30} {Config.NUM_WORKERS:<20}")
    print("-"*80)
    print()
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        Config.BATCH_SIZE,
        Config.NUM_WORKERS
    )
    
    # Initialize model
    model = MNISTNet(dropout_rate=Config.DROPOUT_RATE).to(Config.DEVICE)
    
    # Define optimizer and loss function
    optimizer = optim.SGD(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_accuracy = 0
    
    # Write header to log file
    header = "\nEpoch    Train Loss  Train Acc%  Valid Loss  Valid Acc%"
    write_log(header)
    write_log("-" * 55)
    
    print("\nEpoch    Train Loss  Train Acc%  Valid Loss  Valid Acc%")
    print("-" * 55)
    
    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss, train_acc = train_epoch(model, Config.DEVICE, train_loader, optimizer, criterion)
        
        # Validate
        val_loss, val_accuracy = validate(model, Config.DEVICE, val_loader, criterion)
        
        # Format metrics
        metrics = f'{epoch:3d}      {train_loss:.4f}     {train_acc:.2f}     {val_loss:.4f}     {val_accuracy:.2f}'
        
        # Print metrics
        print(metrics)
        write_log(metrics)
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            save_msg = f'Saved model with accuracy: {val_accuracy:.2f}%'
            print(save_msg)
            write_log(save_msg)

if __name__ == '__main__':
    main() 