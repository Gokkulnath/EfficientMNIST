import torch
from torch import nn
import math

def calculate_rf_info(model):
    """Calculate receptive field info for each layer"""
    def calc_rf_and_jump(kernel_size, stride, prev_rf, prev_j):
        """
        Calculate receptive field and jump for a layer
        RF = prev_RF + (kernel-1) * prev_jump
        jump = prev_jump * stride
        """
        curr_rf = prev_rf + ((kernel_size - 1) * prev_j)
        curr_j = prev_j * stride
        return curr_rf, curr_j

    rf_info = []
    current_rf = 1
    current_j = 1
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            current_rf, current_j = calc_rf_and_jump(
                module.kernel_size[0],
                module.stride[0],
                current_rf,
                current_j
            )
            rf_info.append((name, current_rf))
        elif isinstance(module, nn.MaxPool2d):
            current_rf, current_j = calc_rf_and_jump(
                module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0],
                module.stride if isinstance(module.stride, int) else module.stride[0],
                current_rf,
                current_j
            )
            rf_info.append((name, current_rf))
    
    return rf_info

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(model, input_size=(1, 1, 28, 28)):
    """Print model summary with receptive field information"""
    device = next(model.parameters()).device
    x = torch.rand(1, *input_size[1:]).to(device)
    
    print("\n" + "="*80)
    print(f"Model Summary for {model.__class__.__name__}")
    print("="*80)
    
    total_params = count_parameters(model)
    print(f"\nTotal Trainable Parameters: {total_params:,}")
    print("-"*80)
    
    print("\nLayer Details:")
    print("-"*80)
    print(f"{'Layer':<40} {'Output Shape':<20} {'Params':<10}")
    print("-"*80)
    
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__.__name__)
            module_idx = len(summary)
            
            m_key = f"{class_name}-{module_idx+1}"
            summary[m_key] = {
                "output_shape": list(output.shape),
                "params": sum(p.numel() for p in module.parameters() if p.requires_grad),
            }
            
        if not isinstance(module, nn.Sequential) and \
           not isinstance(module, nn.ModuleList) and \
           not (module == model):
            hooks.append(module.register_forward_hook(hook))
            
    # Create summary dict
    summary = {}
    hooks = []
    
    # Register hooks
    model.apply(register_hook)
    
    # Make a forward pass
    model(x)
    
    # Remove these hooks
    for h in hooks:
        h.remove()
    
    # Print summary
    for layer in summary:
        output_shape = str(summary[layer]["output_shape"])
        params = summary[layer]["params"]
        print(f"{layer:<40} {output_shape:<20} {params:<10}")
    
    print("-"*80)
    print(f"Total params: {total_params:,}")
    print("="*80)
    return summary 