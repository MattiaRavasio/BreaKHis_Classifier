import torch
import torch.nn as nn
from torchvision.models import resnet34

class BasicConv2d(nn.Module):

    
    def __init__(self, dim_in , dim_out, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        
        self.conv = nn.Conv2d(dim_in,
                              dim_out,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding) 
        
        self.bn = nn.BatchNorm2d(dim_out,
                                 eps=0.001, 
                                 momentum=0.1, 
                                 affine=True)
        
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Inception_Skip_Block(nn.Module):

    def __init__(self, scale=0.15):
        super(Inception_Skip_Block, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(512, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(512, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(512, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1),
            BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(128, 512, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class CustomNet(nn.Module):

    def __init__(self):
        
        super().__init__()
        
        resnet = resnet34(pretrained = True)
        
        for param in resnet.parameters():
            param.requires_grad = False
            
        self.conv1 = resnet.conv1
        
        self.bn1 = resnet.bn1
        
        self.relu = resnet.relu
        
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        
        self.layer2 = resnet.layer2
        
        self.layer3 = resnet.layer3
        
        self.layer4 = resnet.layer4
        
        self.Incep_Skip = Inception_Skip_Block(scale=0.15)
        
        self.bn2 = nn.BatchNorm2d(512,eps=0.001, momentum=0.1, affine=True)
        
        self.avgpool = resnet.avgpool
        
        n_inputs = resnet.fc.in_features
        
        self.fc_layer = nn.Sequential(
                            nn.Dropout(0.5),
                            nn.Linear(n_inputs, 2)
                            )
                         
        
        self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')


    def forward(self, x):
        
        dim = x.shape[0]
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.Incep_Skip(x)
        x = self.bn2(x)
        x = self.avgpool(x)
        
        flattened_conv_features = x.view(dim,-1)
        model_output = self.fc_layer(flattened_conv_features)

        return model_output