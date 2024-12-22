import torch

class Config:
    # Data parameters
    BATCH_SIZE = 512
    NUM_WORKERS = 4
    
    # Training parameters
    EPOCHS = 20
    LEARNING_RATE = 0.03
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    
    # Model parameters
    DROPOUT_RATE = 0.05
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Random seed for reproducibility
    SEED = 42 