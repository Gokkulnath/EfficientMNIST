import torch
from models.model import MNISTNet
from utils.data_utils import get_data_loaders
from config import Config

def test():
    # Load data
    _, _, test_loader = get_data_loaders(Config.BATCH_SIZE, Config.NUM_WORKERS)
    
    # Load model
    model = MNISTNet(dropout_rate=Config.DROPOUT_RATE).to(Config.DEVICE)
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Test
    model.eval()
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(Config.DEVICE), target.to(Config.DEVICE)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    test() 