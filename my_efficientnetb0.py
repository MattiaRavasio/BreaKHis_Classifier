import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0


class MyEfficentnetB0(nn.Module):
    def __init__(self):
        
        super().__init__()

        eff = efficientnet_b0(pretrained = True)
        
        for param in eff.parameters():
            param.requires_grad = False
            
        self.features = eff.features
        
        self.avgpool = eff.avgpool
        
        self.fc_layers = nn.Sequential(
                        nn.Linear(1280, 2)
                        )
        
        self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        
        xx = self.features(x)
        conv_features = self.avgpool(xx)
        
        flattened_conv_features = conv_features.view(x.shape[0],-1)
        model_output = self.fc_layers(flattened_conv_features)

        return model_output