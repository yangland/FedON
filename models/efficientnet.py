import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=10, dataset='cifar10'):
        super(EfficientNetB0, self).__init__()
        # Load pretrained EfficientNet-B0 and remove the classifier
        self.base = efficientnet_b0(weights=None)  # or use weights=EfficientNet_B0_Weights.DEFAULT for pretrained
        self.base.classifier = nn.Identity()  # remove the original classifier

        # Replace with custom classifier
        self.classifier = nn.Sequential(
            nn.Linear(1280, num_classes)  # 1280 is the output feature size of EfficientNet-B0
        )

    def forward(self, x):
        x = self.base.features(x)
        x = self.base.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
