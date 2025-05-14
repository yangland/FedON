from models.mnist_6l import Mnist_6L
from models.convnet import ConvNet
from models.cifar10_18l import Cifar10_18L 
from models.resnet import ResNet18
from models.vgg import VGG13, VGG16, VGG11
from models.efficientnet import EfficientNetB0
import torchvision.models as models
import torch.nn as nn

def get_model(model):
    # Dictionary mapping model names to their corresponding classes
    model_dict = {    
        "ConvNet": ConvNet, # good
        "Mnist_6L": Mnist_6L,
        "Cifar10_18L": Cifar10_18L,
        "ResNet18": ResNet18, # good
        "VGG13": VGG13,
        "VGG11": VGG11,
        "VGG16": VGG16,
        "EfficientNetB0": EfficientNetB0,
        "MobileNetV2": MobileNetV2, # good
        "SqueezeNet": SqueezeNet,
    }

    # Check if the model name exists in the dictionary
    if model in model_dict:
        return model_dict[model]  # Return the model class as a tuple
    else:
        raise KeyError(f"Model '{model}' not found. Available models are: {', '.join(model_dict.keys())}")



# Example MobileNetV2 wrapper
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10, dataset="cifar10"):
        super(MobileNetV2, self).__init__()
        self.model = models.mobilenet_v2(pretrained=False)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

    def forward(self, x):
        return self.model(x)

# Example SqueezeNet wrapper
class SqueezeNet(nn.Module):
    def __init__(self, num_classes=10, dataset="cifar10"):
        super(SqueezeNet, self).__init__()
        self.model = models.squeezenet1_0(pretrained=False)
        self.model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        self.model.num_classes = num_classes

    def forward(self, x):
        return self.model(x)
