import torch 
import torch.nn as nn
input_image=torch.randn(1,1,28,28)
conv_layer=nn.Conv2d(1,16,kernel_size=3,stride=1,padding=1)
output=conv_layer(input_image)
print(f"Input Shape: {input_image.shape}")
print(f"Output Shape: {output.shape}")