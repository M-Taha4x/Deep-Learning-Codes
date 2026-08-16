import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
transform=transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor()
])
class MyASLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(128,64)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(p=0.5)
        self.fc2=nn.Linear(64,26)
        
    def forward(self,x):
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        x=self.relu(x)
        x=self.dropout(x)
        x=self.fc2(x)
        return x
    
model = MyASLModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

dummy_input = torch.randn(4, 1, 8, 16)  
output = model(dummy_input)
print("Output shape:", output.shape)