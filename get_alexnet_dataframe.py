import os
import pandas as pd
import numpy as np
import torch.utils
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from data_trasforms import get_fundamental_transforms


def get_frame_from_data_alex(custom, path, inp_size, dataset_mean, dataset_std, which_set ):
    test_dataset = ImageFolder(os.path.join(path, which_set), transform=get_fundamental_transforms(inp_size, dataset_mean, dataset_std))
    test_loader = torch.utils.data.DataLoader(
               test_dataset, batch_size=1, shuffle=True, 
               )

    features_dict = {'feature': [], 'class':[]}

    custom.eval()

    for _, batch in enumerate(test_loader):
            input_data, target_data = Variable(batch[0]), Variable(batch[1]) ### use this if model run on CPU
            #########input_data, target_data = Variable(batch[0]).cuda(), Variable(batch[1]).cuda()   

            conv_features = custom.conv_layers(input_data)
            flattened_features = conv_features.view(input_data.shape[0],-1)
        
            features_dict['feature'].append(np.array(flattened_features).flatten())
            features_dict['class'].append(np.array(target_data)[0])

    features_dataframe = pd.DataFrame(features_dict)
    
    custom.eval()
    
    return features_dataframe