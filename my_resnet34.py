import torch
import torch.nn as nn
from torchvision.models import resnet34


class MyResNet34(nn.Module):
    def __init__(self):
        
        super().__init__()

        resnet = resnet34(pretrained = True)
        
        for param in resnet.parameters():
            param.requires_grad = False
            
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        self.bn1= nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
                                    
        
        self.layer1 = resnet.layer1
        
        self.layer2 = resnet.layer2
        
        self.layer3 = resnet.layer3
        
        self.layer4 = resnet.layer4
        
        self.avgpool = resnet.avgpool
    
        
        n_inputs = resnet.fc.in_features
        
        self.fc_layers = nn.Sequential(nn.Linear(n_inputs, 256), nn.ReLU(), 
                         nn.Linear(256, 2))
        
        self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        
        xx = self.conv1(x)
        xx = self.bn1(xx)
        xx = self.relu(xx)
        xx = self.maxpool(xx)
        
        xx = self.layer1(xx)
        xx = self.layer2(xx)
        xx = self.layer3(xx)
        xx = self.layer4(xx)
        conv_features = self.avgpool(xx)
        
        flattened_conv_features = conv_features.view(x.shape[0],-1)
        model_output = self.fc_layers(flattened_conv_features)

        return model_output
