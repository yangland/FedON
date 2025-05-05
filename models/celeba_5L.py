import torch.nn as nn
import timm

# works for 64 by 64 images
class Celeba_5L(nn.Module):
    def __init__(self, num_classes=10):
        super(Celeba_5L, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=False)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        enet_out_size = 1280
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enet_out_size, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output
    

if __name__ == '__main__':
    model = Celeba_5L()
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("pytorch_total_params", pytorch_total_params)
    # print(model) 5301358