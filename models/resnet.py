import torch
import torch.nn as nn
import torch.nn.functional as F

channel_dict =  {
    "cifar10": 3,
    "cinic10": 3,
    "cifar100": 3,
    "imagenette": 3,
    "SVHN": 3,
    "mnist": 1,
    "fmnist": 1,
    "femnist": 1,
}

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d  # fallback

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=10, norm_layer=None, dataset='cifar10'):
        super(ResNet18, self).__init__()
        if norm_layer is None:
            # Use GroupNorm by default
            self.norm_layer = lambda num_channels: nn.GroupNorm(num_groups=16, num_channels=num_channels)
        else:
            self.norm_layer = norm_layer

        channel = channel_dict.get(dataset, 3)
        self.in_planes = 64

        self.conv1 = nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self.norm_layer(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride, self.norm_layer))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def get_feature(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out



def default_norm_layer(num_channels, num_groups=16):
    return nn.GroupNorm(num_groups=min(num_groups, num_channels), num_channels=num_channels)

    
class ResNet20(nn.Module):
    def __init__(self, num_classes=100, norm_layer=None, dataset='cifar100'):
        super(ResNet20, self).__init__()
        if norm_layer is None:
            norm_layer = default_norm_layer
        self.norm_layer = norm_layer

        channel = channel_dict.get(dataset, 3)
        self.in_planes = 16

        self.conv1 = nn.Conv2d(channel, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self.norm_layer(16)
        self.layer1 = self._make_layer(16, 3, stride=1)
        self.layer2 = self._make_layer(32, 3, stride=2)
        self.layer3 = self._make_layer(64, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s, self.norm_layer))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.linear(out)



class ResNet32(nn.Module):
    def __init__(self, num_classes=100, norm_layer=None, dataset='cifar100'):
        super(ResNet32, self).__init__()
        if norm_layer is None:
            norm_layer = default_norm_layer
        self.norm_layer = norm_layer

        channel = channel_dict.get(dataset, 3)
        self.in_planes = 16

        self.conv1 = nn.Conv2d(channel, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self.norm_layer(16)
        self.layer1 = self._make_layer(16, 5, stride=1)
        self.layer2 = self._make_layer(32, 5, stride=2)
        self.layer3 = self._make_layer(64, 5, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s, self.norm_layer))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.linear(out)


class ResNet44(nn.Module):
    def __init__(self, num_classes=100, norm_layer=None, dataset='cifar100'):
        super(ResNet44, self).__init__()
        if norm_layer is None:
            norm_layer = default_norm_layer
        self.norm_layer = norm_layer
        
        channel = channel_dict.get(dataset, 3)
        self.in_planes = 16

        self.conv1 = nn.Conv2d(channel, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self.norm_layer(16)
        self.layer1 = self._make_layer(16, 7, stride=1)
        self.layer2 = self._make_layer(32, 7, stride=2)
        self.layer3 = self._make_layer(64, 7, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s, self.norm_layer))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.linear(out)



class ResNet14_Light(nn.Module):
    def __init__(self, num_classes=100, norm_layer=None, dataset='cifar100'):
        super(ResNet14_Light, self).__init__()
        if norm_layer is None:
            self.norm_layer = lambda num_channels: nn.GroupNorm(num_groups=16, num_channels=num_channels)
        else:
            self.norm_layer = norm_layer

        channel = channel_dict.get(dataset, 3)
        self.in_planes = 32  # Smaller model

        self.conv1 = nn.Conv2d(channel, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self.norm_layer(32)
        self.layer1 = self._make_layer(32, 2, stride=1)
        self.layer2 = self._make_layer(64, 2, stride=2)
        self.layer3 = self._make_layer(128, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(128, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s, self.norm_layer))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet8_Light(nn.Module):
    def __init__(self, num_classes=10, norm_layer=None, dataset='fmnist'):
        super(ResNet8_Light, self).__init__()
        if norm_layer is None:
            norm_layer = lambda num_channels: nn.GroupNorm(num_groups=16, num_channels=num_channels)
        self.norm_layer = norm_layer
        in_channels = channel_dict.get(dataset, 1)
        self.in_planes = 16

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self.norm_layer(16)

        # ResNet8 = 3 layers, 2 blocks each
        self.layer1 = self._make_layer(16, 2, stride=1)
        self.layer2 = self._make_layer(32, 2, stride=2)  # Downsample
        self.layer3 = self._make_layer(64, 2, stride=2)  # Downsample

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s, self.norm_layer))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def bn_to_gn(module, num_groups=16):
    """
    Recursively convert all BatchNorm2d layers to GroupNorm.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            gn = nn.GroupNorm(num_groups=min(num_groups, num_channels), num_channels=num_channels)
            setattr(module, name, gn)
        else:
            bn_to_gn(child, num_groups)
