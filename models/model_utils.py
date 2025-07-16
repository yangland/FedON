from models.mnist_6l import Mnist_6L
from models.convnet import ConvNet
from models.cifar10_18l import Cifar10_18L 
from models.resnet import ResNet18, ResNet14_Light, ResNet8_Light, ResNet20, ResNet8_Light_one_channel, ResNet32, ResNet44, bn_to_gn
from models.vgg import VGG13, VGG16, VGG11, VGG13_32
from models.efficientnet import EfficientNetB0
import torchvision.models as models
import torch.nn as nn
from torchvision.models.squeezenet import Fire
import torch.nn.functional as F
import inspect
import torch

def get_model(model_name):
    model_dict = {
        "ConvNet": ConvNet,
        "Mnist_6L": Mnist_6L,
        "Cifar10_18L": Cifar10_18L,
        "VGG11": VGG11,
        "VGG13": VGG13,
        "VGG13_32": VGG13_32, # wrks for cifar10
        "VGG16": VGG16,
        "EfficientNetB0": EfficientNetB0, 
        "MobileNetV2": MobileNetV2,
        "MobileNetV2_32": MobileNetV2_32, # not working for cifar10
        "SqueezeNet": SqueezeNet,
        "SqueezeNet_32": SqueezeNet1_1_32, # works for cifar10
        "LeNet5": LeNet5,
        "SimpleCNN_28": SimpleCNN_28,
        "SimpleCNN_32": SimpleCNN_32, # works for cifar10
        "ResNet18": ResNet18, # works for cifar10
        "ResNet14_Light": ResNet14_Light, # for cifar100 works
        "ResNet8_Light": ResNet8_Light,
        "ResNet8_Light_one_channel":ResNet8_Light_one_channel,
        "ResNet20": ResNet20
    }

    if model_name not in model_dict:
        raise KeyError(f"Model '{model_name}' not found. Available: {', '.join(model_dict.keys())}")

    model_class = model_dict[model_name]

    def model_wrapper(**kwargs):
        sig = inspect.signature(model_class.__init__)
        valid_args = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return model_class(**valid_args)

    return model_wrapper


# Example MobileNetV2 wrapper
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10, use_groupnorm=True):
        super(MobileNetV2, self).__init__()
        self.model = models.mobilenet_v2(weights=None)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)
        
        # Replace all BatchNorm with GroupNorm
        if use_groupnorm:
            bn_to_gn(self.model)
    
    def forward(self, x):
        return self.model(x)


class MobileNetV2_32(nn.Module):
    def __init__(self, num_classes=10, use_groupnorm=True):
        super(MobileNetV2_32, self).__init__()
        self.model = models.mobilenet_v2(weights=None)

        # Modify the first layer for smaller input (e.g., CIFAR10)
        self.model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)

        # Modify classifier
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

        # Replace all BatchNorm with GroupNorm
        if use_groupnorm:
            bn_to_gn(self.model)

    def forward(self, x):
        return self.model(x)


class MobileNetV2_28(nn.Module):
    def __init__(self, num_classes=10, use_groupnorm=True):
        super(MobileNetV2_28, self).__init__()
        self.model = models.mobilenet_v2(weights=None)
        
        # Adjust first convolution: in_channels=1 for grayscale, smaller input
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Modify the classifier to output the desired number of classes
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)
 
        # Replace all BatchNorm with GroupNorm
        if use_groupnorm:
            bn_to_gn(self.model) 
        
    def forward(self, x):
        return self.model(x)


# Example SqueezeNet wrapper
class SqueezeNet(nn.Module):
    def __init__(self, num_classes=10, use_groupnorm=True):
        super(SqueezeNet, self).__init__()
        self.model = models.squeezenet1_0(weights=None)
        self.model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        self.model.num_classes = num_classes
        # Replace all BatchNorm with GroupNorm
        if use_groupnorm:
            bn_to_gn(self.model)

    def forward(self, x):
        return self.model(x)



class SqueezeNet1_1_32(nn.Module):
    def __init__(self, num_classes=10):
        super(SqueezeNet1_1_32, self).__init__()

        # Load base model without pretrained weights
        self.model = models.squeezenet1_1(weights=None)

        # Modify the first conv layer to avoid aggressive downsampling
        self.model.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        # Remove the maxpool after the first conv layer (downsampling too early)
        self.model.features[1] = nn.Identity()

        # Add adaptive pooling before classifier to ensure fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Replace the classifier
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            self.adaptive_pool
        )

        self.num_classes = num_classes

    def forward(self, x):
        x = self.model(x)
        return x.view(x.size(0), -1)  # Flatten to (batch_size, num_classes)

    
    

class SqueezeNet_28(nn.Module):
    def __init__(self, num_classes=10):
        super(SqueezeNet_28, self).__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            Fire(64, 16, 64, 64),   # Fire2, inplanes=64
            Fire(128, 16, 64, 64),  # Fire3
            Fire(128, 32, 128, 128),# Fire4
            nn.MaxPool2d(kernel_size=2, stride=2),

            Fire(256, 32, 128, 128),# Fire5
            Fire(256, 48, 192, 192),# Fire6
            Fire(384, 48, 192, 192),# Fire7
            Fire(384, 64, 256, 256),# Fire8
            nn.MaxPool2d(kernel_size=2, stride=2),
            Fire(512, 64, 256, 256),# Fire9
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)



class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)     # 28x28 -> 24x24
        self.pool = nn.AvgPool2d(2, 2)                  # 24x24 -> 12x12
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)    # 12x12 -> 8x8
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        x = x.view(-1, 16*4*4)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

class SimpleCNN_28(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN_28, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # (1,28,28) -> (32,28,28)
        self.pool = nn.MaxPool2d(2, 2)                           # -> (32,14,14)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # -> (64,14,14)
        self.pool2 = nn.MaxPool2d(2, 2)                          # -> (64,7,7)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)                    # 64*7*7 = 3136
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # -> (32,14,14)
        x = self.pool2(F.relu(self.conv2(x)))  # -> (64,7,7)
        x = x.view(x.size(0), -1)              # Flatten to (B, 3136)
        x = F.relu(self.fc1(x))
        return self.fc2(x)



class SimpleCNN_32(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN_32, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # (3,32,32) -> (32,32,32)
        self.pool = nn.MaxPool2d(2, 2)                            # -> (32,16,16)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # -> (64,16,16)
        self.pool2 = nn.MaxPool2d(2, 2)                           # -> (64,8,8)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)                     # 64*8*8 = 4096
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))     # -> (32,16,16)
        x = self.pool2(F.relu(self.conv2(x)))    # -> (64,8,8)
        x = x.view(x.size(0), -1)                # Flatten to (B, 4096)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
