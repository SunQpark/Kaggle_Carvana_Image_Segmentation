import os
import numpy as np
import pandas as pd
import pickle as pkl
import gzip
from utils.img2patch import rand_crop

def load_data():
    with gzip.open('inputs/data_patch/data.pkl.gz') as f:
        images, labels = pkl.load(f)
    return images, labels



if __name__ == '__main__':
    id_list = [id.replace('.jpg', '') for id in  os.listdir('inputs/train')]
    
    images = None
    labels = None

    for id in id_list:
        if images is None:
            images, labels = rand_crop(id)
        else:
            images_patch, labels_patch = rand_crop(id)
            images = np.concatenate((images, images_patch), axis=0)
            labels = np.concatenate((labels, labels_patch), axis=0)
    
    num_data = images.shape[0]
    shuffle = np.random.permutation(range(num_data))
    

    with gzip.open('inputs/data_patch/data.pkl.gz', 'wb') as f:
        pkl.dump((images[shuffle], labels[shuffle]), f)
