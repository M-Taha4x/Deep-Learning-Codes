import torch 
import torch.nn as nn
conv_output=torch.randn(1,64,100,100)
pool_layer=nn.MaxPool2d(kernel_size=2,stride=2)
pooled_output=pool_layer(conv_output)
print(f"Input Shape: {conv_output.shape}")
print(f"Output Shape {pooled_output.shape}")
