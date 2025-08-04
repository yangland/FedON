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

class ResNet18_Light(nn.Module):
    def __init__(self, num_classes=10, norm_layer=None, dataset='cifar10'):
        super(ResNet18_Light, self).__init__()
        if norm_layer is None:
            self.norm_layer = lambda c: nn.GroupNorm(num_groups=4, num_channels=c)
        else:
            self.norm_layer = norm_layer

        channel = channel_dict.get(dataset, 3)
        self.in_planes = 32  # LIGHTER

        self.conv1 = nn.Conv2d(channel, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self.norm_layer(32)

        # 4 stages like ResNet18 but with smaller channels
        self.layer1 = self._make_layer(32, 2, stride=1)   # 2 blocks
        self.layer2 = self._make_layer(64, 2, stride=2)   # 2 blocks
        self.layer3 = self._make_layer(96, 2, stride=2)   # 2 blocks
        self.layer4 = self._make_layer(128, 2, stride=2)  # 2 blocks

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(128 * BasicBlock.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s, self.norm_layer))
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
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return out
    

# ResNet18_Light for 1-channel input (e.g., FMNIST)
class ResNet18_Light_one_channel(nn.Module):
    def __init__(self, num_classes=10, norm_layer=None, dataset='fmnist'):
        super(ResNet18_Light_one_channel, self).__init__()
        if norm_layer is None:
            self.norm_layer = lambda c: nn.GroupNorm(num_groups=4, num_channels=c)
        else:
            self.norm_layer = norm_layer

        channel = channel_dict.get(dataset, 1)  # 1 channel for grayscale
        self.in_planes = 32  # LIGHTER

        self.conv1 = nn.Conv2d(channel, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self.norm_layer(32)

        # 4 stages like ResNet18 but with smaller channels
        self.layer1 = self._make_layer(32, 2, stride=1)   # 2 blocks
        self.layer2 = self._make_layer(64, 2, stride=2)   # 2 blocks
        self.layer3 = self._make_layer(96, 2, stride=2)   # 2 blocks
        self.layer4 = self._make_layer(128, 2, stride=2)  # 2 blocks

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(128 * BasicBlock.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s, self.norm_layer))
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
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
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


# ResNet8 Light for FMNIST
class ResNet8_Light_one_channel(nn.Module):
    def __init__(self, num_classes=10, norm_layer=None, dataset='fmnist'):
        super(ResNet8_Light_one_channel, self).__init__()
        if norm_layer is None:
            # Use GroupNorm with num_groups=4 for small channels (e.g., 16, 32, 64)
            norm_layer = lambda num_channels: nn.GroupNorm(num_groups=4, num_channels=num_channels)

        self.norm_layer = norm_layer
        in_channels = channel_dict.get(dataset, 1)
        self.in_planes = 16

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self.norm_layer(16)

        # ResNet8 = 3 layers, each with 2 blocks
        self.layer1 = self._make_layer(16, 2, stride=1)
        self.layer2 = self._make_layer(32, 2, stride=2)  # Downsampling
        self.layer3 = self._make_layer(64, 2, stride=2)  # Downsampling

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
        out = F.relu(self.bn1(self.conv1(x)))  # Initial conv
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    

class ResNet34_Light(nn.Module):
    def __init__(self, num_classes=10, norm_layer=None, dataset='imagenette'):
        super(ResNet34_Light, self).__init__()
        if norm_layer is None:
            self.norm_layer = lambda c: nn.GroupNorm(num_groups=4, num_channels=c)
        else:
            self.norm_layer = norm_layer

        channel = channel_dict.get(dataset, 3)
        self.in_planes = 32  # LIGHTER than standard ResNet34

        self.conv1 = nn.Conv2d(channel, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self.norm_layer(32)

        # ResNet34 block config: [3, 4, 6, 3] with reduced channels
        self.layer1 = self._make_layer(32, 3, stride=1)
        self.layer2 = self._make_layer(64, 4, stride=2)
        self.layer3 = self._make_layer(96, 6, stride=2)
        self.layer4 = self._make_layer(128, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(128 * BasicBlock.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s, self.norm_layer))
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
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return out


class ResNet50_Light(nn.Module):
    def __init__(self, num_classes=100, norm_layer=None, dataset='cifar100'):
        super(ResNet50_Light, self).__init__()

        if norm_layer is None:
            self.norm_layer = lambda c: nn.GroupNorm(num_groups=4, num_channels=c)
        else:
            self.norm_layer = norm_layer

        channel = channel_dict.get(dataset, 3)
        self.in_planes = 32  # lighter starting point

        self.conv1 = nn.Conv2d(channel, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self.norm_layer(32)

        # Define the layers with Bottleneck blocks (3, 4, 6, 3)
        self.layer1 = self._make_layer(32, 3, stride=1)
        self.layer2 = self._make_layer(64, 4, stride=2)
        self.layer3 = self._make_layer(96, 6, stride=2)
        self.layer4 = self._make_layer(128, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(128 * Bottleneck.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(Bottleneck(self.in_planes, planes, s, self.norm_layer))
            self.in_planes = planes * Bottleneck.expansion
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
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = lambda c: nn.GroupNorm(num_groups=4, num_channels=c)

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = norm_layer(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * self.expansion)
            )

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

def bn_to_gn(module, max_groups=16):
    # Make a static copy of the children to avoid modifying the dictionary during iteration
    children = list(module.named_children())

    for name, child in children:
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features

            # Safely remove the old module (BatchNorm) and its buffers
            delattr(module, name)

            # Replace with compatible GroupNorm
            for num_groups in reversed(range(1, max_groups + 1)):
                if num_channels % num_groups == 0:
                    gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
                    setattr(module, name, gn)
                    break
            else:
                raise ValueError(f"Cannot find valid num_groups for GroupNorm with num_channels={num_channels}")
        else:
            # Recursively apply to child modules
            bn_to_gn(child, max_groups)

