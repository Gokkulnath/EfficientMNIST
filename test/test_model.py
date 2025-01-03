import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import pytest
from models.model import MNISTNet
from utils.model_utils import count_parameters
from utils.data_utils import get_data_loaders
from config import Config

def test_parameter_count():
    """Test that model has less than 20k parameters"""
    model = MNISTNet()
    param_count = count_parameters(model)
    assert param_count < 20000, f"Model has {param_count:,} parameters, should be less than 20,000"

def test_has_batchnorm():
    """Test that model uses batch normalization"""
    model = MNISTNet()
    has_batchnorm = False
    
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            has_batchnorm = True
            break
    
    assert has_batchnorm, "Model should use BatchNormalization"

def test_has_gap():
    """Test that model uses Global Average Pooling"""
    model = MNISTNet()
    has_gap = False
    
    for module in model.modules():
        if isinstance(module, nn.AvgPool2d):
            # Check if it's used as GAP (kernel size matches final feature map size)
            if module.kernel_size == 2:  # We know our final feature map is 2x2
                has_gap = True
                break
    
    assert has_gap, "Model should use Global Average Pooling"

def test_final_layer():
    """Test that model ends with a fully connected layer"""
    model = MNISTNet()
    layers = list(model.modules())
    
    # Find the last non-Dropout layer
    final_layer = None
    for layer in reversed(layers):
        if not isinstance(layer, nn.Dropout):
            final_layer = layer
            break
    
    assert isinstance(final_layer, nn.Linear), "Final layer should be Linear (Fully Connected)"
    assert final_layer.out_features == 10, "Final layer should output 10 classes"

def test_architecture_sequence():
    """Test the proper sequence of layers"""
    model = MNISTNet()
    
    # Check if conv layers are followed by batchnorm
    conv_followed_by_bn = True
    prev_layer = None
    
    for module in model.modules():
        if isinstance(prev_layer, nn.Conv2d):
            if not isinstance(module, (nn.BatchNorm2d, nn.Sequential)):
                conv_followed_by_bn = False
                break
        prev_layer = module
    
    assert conv_followed_by_bn, "Each Conv2d should be followed by BatchNorm2d"


if __name__ == "__main__":
    pytest.main([__file__]) 