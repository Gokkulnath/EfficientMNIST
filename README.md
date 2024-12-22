# Efficient MNIST Classifier

A highly efficient PyTorch implementation of a CNN classifier for the MNIST dataset, achieving >99.4% validation accuracy with less than 20k parameters.

## Architecture Highlights

- No padding in convolutions for parameter efficiency
- Batch Normalization after every convolution
- Global Average Pooling to reduce parameters
- Efficient channel progression
- Dropout for regularization

## Model Architecture 

Input Image (1x28x28)
│
├── Conv1a (8 channels, 3x3) -> BN -> ReLU
│ └── Conv1b (16 channels, 3x3) -> BN -> ReLU -> MaxPool(2x2) -> Dropout
│
├── Conv2a (16 channels, 3x3) -> BN -> ReLU
│ └── Conv2b (32 channels, 3x3) -> BN -> ReLU -> MaxPool(2x2) -> Dropout
│
├── Conv3 (32 channels, 3x3) -> BN -> ReLU -> Dropout
│
├── Global AvgPool (2x2)
└── FC Layer (10 outputs)


## Requirements

```bash
pip install -r requirements.txt
```

## Usage

### Training
```bash
python train.py
```

The training script:
- Uses MNIST dataset (automatically downloaded)
- Saves the best model as 'best_model.pth'
- Logs training progress to 'training_logs.log'
- Prints model summary and training parameters


### Testing

```bash
pytest test/test_model.py -v
```

Tests verify that the model:
- Has less than 20k parameters
- Uses Batch Normalization
- Uses Global Average Pooling
- Achieves >99.4% validation accuracy

## Configuration

Key parameters in `config.py`:
```python
BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 0.03
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
DROPOUT_RATE = 0.05
```

## Project Structure

├── models/
│ └── model.py # Model architecture
├── utils/
│ ├── data_utils.py # Data loading utilities
│ └── model_utils.py # Model helper functions
├── test/
│ └── test_model.py # Model tests
├── config.py # Configuration parameters
├── train.py # Training script
└── requirements.txt # Dependencies

# Training Logs

```
Epoch    Train Loss  Train Acc%  Valid Loss  Valid Acc%
[2024-12-22 22:55:59] -------------------------------------------------------
[2024-12-22 22:56:10]   1      0.4290     89.74     0.0872     97.68
[2024-12-22 22:56:10] Saved model with accuracy: 97.68%
[2024-12-22 22:56:22]   2      0.0679     98.26     0.0512     98.68
[2024-12-22 22:56:22] Saved model with accuracy: 98.68%
[2024-12-22 22:56:34]   3      0.0494     98.67     0.0401     98.89
[2024-12-22 22:56:34] Saved model with accuracy: 98.89%
[2024-12-22 22:56:45]   4      0.0404     98.94     0.0382     98.88
[2024-12-22 22:56:57]   5      0.0340     99.11     0.0350     98.95
[2024-12-22 22:56:57] Saved model with accuracy: 98.95%
[2024-12-22 22:57:09]   6      0.0302     99.18     0.0317     99.16
[2024-12-22 22:57:09] Saved model with accuracy: 99.16%
[2024-12-22 22:57:20]   7      0.0270     99.30     0.0277     99.26
[2024-12-22 22:57:20] Saved model with accuracy: 99.26%
[2024-12-22 22:57:32]   8      0.0257     99.32     0.0295     99.24
[2024-12-22 22:57:44]   9      0.0225     99.41     0.0305     99.20
[2024-12-22 22:57:55]  10      0.0216     99.45     0.0255     99.26
[2024-12-22 22:58:07]  11      0.0199     99.51     0.0251     99.27
[2024-12-22 22:58:07] Saved model with accuracy: 99.27%
[2024-12-22 22:58:20]  12      0.0189     99.57     0.0262     99.22
[2024-12-22 22:58:32]  13      0.0177     99.55     0.0252     99.29
[2024-12-22 22:58:32] Saved model with accuracy: 99.29%
[2024-12-22 22:58:44]  14      0.0169     99.61     0.0233     99.34
[2024-12-22 22:58:44] Saved model with accuracy: 99.34%
[2024-12-22 22:58:55]  15      0.0152     99.64     0.0227     99.40
[2024-12-22 22:58:55] Saved model with accuracy: 99.40%
[2024-12-22 22:59:08]  16      0.0148     99.66     0.0223     99.32
[2024-12-22 22:59:20]  17      0.0144     99.65     0.0227     99.31
[2024-12-22 22:59:32]  18      0.0139     99.71     0.0255     99.26
[2024-12-22 22:59:44]  19      0.0140     99.66     0.0240     99.32
[2024-12-22 22:59:55]  20      0.0135     99.69     0.0217     99.43
[2024-12-22 22:59:56] Saved model with accuracy: 99.43%
```

## CI/CD

GitHub Actions workflow:
- Trains model on every push/PR
- Runs all tests
- Verifies accuracy requirements
- Saves trained model as artifact

## Performance

- Parameters: <20k
- Validation Accuracy: >99.4%
- Training Time: ~5 minutes (CPU)
- Memory Efficient: ~100MB peak memory

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch team for the excellent framework
- MNIST dataset creators
- Open source community
