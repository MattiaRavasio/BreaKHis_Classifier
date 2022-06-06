import glob
import os
from typing import Tuple
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler

def compute_mean_and_std(df):

    scaler = StandardScaler()

    for i in df['filename']:
        path = np.array(Image.open(os.path.join('archive/BreaKHis_v1',i)))
        image = path / 255.0
        image = np.reshape(image, (-1, 1))
        scaler.partial_fit(image)

    mean = scaler.mean_
    std = np.sqrt(scaler.var_)

    return mean, std
