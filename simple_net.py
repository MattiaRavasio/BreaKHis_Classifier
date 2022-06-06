import torch
import torch.nn as nn


class SimpleNet(nn.Module):

    def __init__(self):
        
        super().__init__()

        conv1 = nn.Conv2d(3,32,5, padding='same')
        #batch1 = nn.BatchNorm2d(10)
        pool1 = nn.MaxPool2d(3)
        relu1 = nn.ReLU()
        
        conv2 = nn.Conv2d(32,64,5)
        #batch2 = nn.BatchNorm2d(20)
        pool2 = nn.MaxPool2d(3)
        relu2 = nn.ReLU()

        conv3 = nn.Conv2d(64,128,5)
        pool3 = nn.MaxPool2d(3)
        relu3 = nn.ReLU() 
        
        lin1 = nn.Linear(238336,100)
        relu4 = nn.ReLU()
        lin2 = nn.Linear(100,15)
        
        self.conv_layers = nn.Sequential(conv1, pool1, relu1 , conv2, pool2, relu2)
        self.fc_layers = nn.Sequential(lin1, relu3, lin2)
        self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')


    def forward(self, x):
        
        conv_features = self.conv_layers(x)
        flattened_conv_features = conv_features.view(x.shape[0],-1)
        model_output = self.fc_layers(flattened_conv_features)

        return model_output
