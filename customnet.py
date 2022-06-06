import torch
import torch.nn as nn

class CustomNet(nn.Module):

    def __init__(self):
        
        super().__init__()
        
        self.layer0= nn.Sequential(nn.Conv2d(3,32,3,stride=1,padding= 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(4,stride=2, padding = 0)
            )
        
        self.layer1= nn.Sequential(
            nn.Conv2d(32,32,3,stride=1 , padding= 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,32,3,stride=1, padding= 1),
            nn.BatchNorm2d(32)
            )
        
        self.res1 = nn.Sequential(nn.BatchNorm2d(32))
        
        self.relu1 = nn.Sequential(nn.ReLU())
        
        self.layer2= nn.Sequential(
            nn.Conv2d(32,64,3,stride=2, padding= 0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,3,stride = 1, padding= 1),
            nn.BatchNorm2d(64)
            )
        
        self.res2 = nn.Sequential(nn.Conv2d(32,64,2,stride=2, padding= 0),nn.BatchNorm2d(64) )
        ####################### problema con stride di 2
        
        self.relu2 = nn.Sequential(nn.ReLU())
        
        self.avgpool = nn.Sequential(nn.AvgPool2d(6, stride=6))
       
        self.fc_layers = nn.Sequential(
            nn.Linear(32256,100),
            nn.ReLU(),
            nn.Linear(100,2)
            )
        
        self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')


    def forward(self, x):
        
        x1 = self.layer0(x)
        x2 = self.layer1(x1) + self.res1(x1)
        x3 = self.relu1(x2)
        x4 = self.layer2(x3) + self.res2(x3)
        x5 = self.relu2(x4)
        x6 = self.avgpool(x5)
        flattened_conv_features = x6.view(x.shape[0],-1)
        model_output = self.fc_layers(flattened_conv_features)

        return model_output
