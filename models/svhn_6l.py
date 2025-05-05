import torch.nn as nn
import torch.nn.functional as F
import torch


class Svhn_6L(nn.Module):
    def __init__(self):
        self.name = "Mnist 6 layers"
        super(Svhn_6L, self).__init__()
        # Convolutional layers
                            #Init_channels, channels, kernel_size, padding) 
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2,2)
        
        # FC layers
        # Linear layer (64x4x4 -> 500)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        
        # Linear Layer (500 -> 10)
        self.fc2 = nn.Linear(500, 10)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.elu(self.conv1(x)))
        x = self.pool(F.elu(self.conv2(x)))
        x = self.pool(F.elu(self.conv3(x)))
        
        # Flatten the image
        x = x.view(-1, 64*4*4)
        x = self.dropout(x)
        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    model=Svhn_6L()
    print(model)
    
    

# class Svhn_6L(nn.Module):
#     def __init__(self):
#         self.name = "Mnist 6 layers"
#         super(Svhn_6L, self).__init__()
#         # hidden layer
#         self.linear1 = nn.Linear(3072, 1024)
#         self.linear2 = nn.Linear(1024, 512)
#         self.linear3 = nn.Linear(512, 256)
#         self.linear4 = nn.Linear(256, 128)
#         self.linear5 = nn.Linear(128, 64)
#         self.linear6 = nn.Linear(64, 32)
#         # output layer
#         self.linear7 = nn.Linear(32, 10)

#     def forward(self, xb):
#         # Flatten the image tensors
#         xb = xb.view(xb.size(0), -1)
#         # Get intermediate outputs using hidden layer
#         out = self.linear1(xb)
#         out = F.relu(out)
#         out = self.linear2(out)
#         out = F.relu(out)
#         out = self.linear3(out)
#         out = F.relu(out)
#         out = self.linear4(out)
#         out = F.relu(out)
#         out = self.linear5(out)
#         out = F.relu(out)
#         out = self.linear6(out)
#         out = F.relu(out)
#         # Get predictions using output layer
#         out = self.linear7(out)
#         return out