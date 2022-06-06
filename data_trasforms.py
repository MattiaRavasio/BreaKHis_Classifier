from typing import Tuple

import torchvision.transforms as transforms
import numpy as np


def get_fundamental_transforms(inp_size, pixel_mean, pixel_std):

    return transforms.Compose(
        [
            transforms.Resize(inp_size),
            transforms.ToTensor(),
            transforms.Normalize(pixel_mean, pixel_std, inplace=True)
        ]
    )

def get_data_augmentation_transforms(inp_size, pixel_mean, pixel_std):

    return transforms.Compose(
        [
            
            transforms.Resize(inp_size),
            transforms.RandomRotation(30),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(pixel_mean, pixel_std, inplace=True)
        ]
    )

def get_efficent_augmentation_transforms(inp_size, pixel_mean, pixel_std):

    return transforms.Compose(
        [
            
            transforms.Resize(inp_size),
            transforms.RandomRotation(30),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ColorJitter (brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomAffine(0,(0.1,0.1)),
            transforms.ToTensor(),
            transforms.Normalize(pixel_mean, pixel_std, inplace=True)
        ]
    )    