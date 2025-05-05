import torch.nn as nn
import torch.nn.functional as F
class Mnist_3L(nn.Module):
    def __init__(self):
        self.name = "Mnist 3 layers"
        super(Mnist_3L, self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    model=Mnist_3L()
    print(model)