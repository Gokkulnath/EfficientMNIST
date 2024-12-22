import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(MNISTNet, self).__init__()
        
        # Initial convolution block
        self.conv1a = nn.Conv2d(1, 8, kernel_size=3)  # output: 8x26x26, RF: 3x3
        self.bn1a = nn.BatchNorm2d(8)
        self.conv1b = nn.Conv2d(8, 16, kernel_size=3)  # output: 16x24x24, RF: 5x5
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)  # output: 16x12x12, RF: 10x10
        
        # Second convolution block
        self.conv2a = nn.Conv2d(16, 16, kernel_size=3)  # output: 16x10x10, RF: 14x14
        self.bn2a = nn.BatchNorm2d(16)
        self.conv2b = nn.Conv2d(16, 32, kernel_size=3)  # output: 16x8x8, RF: 18x18
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)  # output: 16x4x4, RF: 36x36
        
        # Third convolution block
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)  # output: 32x2x2, RF: 44x44
        self.bn3 = nn.BatchNorm2d(32)  # output: 32x2x2, RF: 44x44
        
        # Global Average Pooling and final layers
        self.gap = nn.AvgPool2d(kernel_size=2)  # output: 32x1x1, RF: 44x44 (global)
        self.dropout = nn.Dropout(dropout_rate)  # output: 32x1x1, RF: 44x44 (global)
        self.fc = nn.Linear(32, 10)  # output: 10

    def forward(self, x):
        # Input: 1x28x28
        
        # First block
        x = self.conv1a(x)  # 8x26x26, RF: 3x3
        x = F.relu(x)
        x = self.bn1a(x)
        x = self.conv1b(x)  # 12x24x24, RF: 5x5
        x = F.relu(x)      # 12x24x24, RF: 5x5
        x = self.bn1(x)    # 12x24x24, RF: 5x5
        x = self.pool1(x)  # 8x12x12, RF: 10x10
        x = self.dropout(x) # 8x12x12, RF: 10x10

        # Second block
        x = self.conv2a(x)  # 16x10x10, RF: 14x14
        x = F.relu(x)
        x = self.bn2a(x)
        x = self.conv2b(x)  # 24x8x8, RF: 18x18
        x = F.relu(x)      # 24x8x8, RF: 18x18
        x = self.bn2(x)    # 24x8x8, RF: 18x18
        x = self.pool2(x)  # 16x4x4, RF: 36x36
        x = self.dropout(x) # 16x4x4, RF: 36x36

        # Third block
        x = self.conv3(x)  # 32x2x2, RF: 44x44
        x = F.relu(x)      # 32x2x2, RF: 44x44
        x = self.bn3(x)    # 32x2x2, RF: 44x44
        x = self.dropout(x) # 32x2x2, RF: 44x44

        # GAP and final layers
        x = self.gap(x)         # Global average pooling over 2x2 -> 32x1x1, RF: 44x44 (global)
        x = x.view(-1, 32)      # 32, RF: 44x44 (global)
        x = self.dropout(x)     # 32, RF: 44x44 (global)
        x = self.fc(x)          # 10
        
        return F.log_softmax(x, dim=1)