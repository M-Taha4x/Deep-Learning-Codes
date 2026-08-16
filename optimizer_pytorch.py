import torch
import torch.nn as nn
import torch.optim as optim
model_layer=nn.Linear(10,2)
optimizer=optim.Adam(model_layer.parameters(),lr=0.001)
print(optimizer)