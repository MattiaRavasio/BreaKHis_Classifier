import torch
import torch.nn as nn
from torchvision.models import vgg16


class MyVGGNet16(nn.Module):
    def __init__(self):
        
        super().__init__()

        vgg = vgg16(pretrained = True)
        
        self.conv_layers = vgg.features
        
        for param in vgg.parameters():
            param.requires_grad = False
        
        vgg.classifier[6] = nn.Linear(in_features=4096, out_features=2, bias=True)
        
        self.avgpool = vgg.avgpool
        
        self.fc_layers = vgg.classifier
        
        self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        
        
        conv_features = self.conv_layers(x)
        
        conv_features = self.avgpool(conv_features)
        
        flattened_conv_features = conv_features.view(x.shape[0],-1)
        model_output = self.fc_layers(flattened_conv_features)

        return model_output

