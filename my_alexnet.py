import torch
import torch.nn as nn
from torchvision.models import alexnet


class MyAlexNet(nn.Module):
    def __init__(self):
        
        super().__init__()

        alex = alexnet(pretrained = True)
        
        self.conv_layers = alex.features
        
        for param in alex.parameters():
            param.requires_grad = False
        
        alex.classifier[6] = nn.Linear(4096, 2)
        
        self.conv_layers = nn.Sequential(alex.features[0],
                                         alex.features[1],
                                         alex.features[2],
                                         alex.features[3],
                                         alex.features[4],
                                         alex.features[5],
                                         alex.features[6],
                                         alex.features[7],
                                         alex.features[8],
                                         alex.features[9],
                                         alex.features[10],
                                         alex.features[11],
                                         alex.features[12],
                                         alex.avgpool)
        
        self.fc_layers = alex.classifier
        self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        
        conv_features = self.conv_layers(x)
        flattened_conv_features = conv_features.view(x.shape[0],-1)
        model_output = self.fc_layers(flattened_conv_features)

        return model_output
