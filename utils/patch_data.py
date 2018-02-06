import os
import numpy as np
import pickle as pkl
import gzip
from utils.img2patch import rand_crop


class InputImage:
    def __init__(self, order, num_images_per_batch):
        self.order = order  # the order of batch
        self.num_images_per_batch = num_images_per_batch
        self.make_patch()

    def make_patch(self):
        id_list = [id.replace('.jpg', '') for id in os.listdir('inputs/train')]

        images = None
        labels = None
        cnt = 0
        for i, id in enumerate(id_list):
            if i >= self.order * self.num_images_per_batch:
                if images is None:
                    images, labels = rand_crop(id)
                else:
                    images_patch, labels_patch = rand_crop(id)
                    images = np.concatenate((images, images_patch), axis=0)
                    labels = np.concatenate((labels, labels_patch), axis=0)
                cnt += 1
                if cnt == self.num_images_per_batch:
                    break
        num_data = images.shape[0]
        shuffle = np.random.permutation(range(num_data))
        images = images[shuffle]
        labels = labels[shuffle]

        return images, labels

        '''
        num_patches_per_batch = 10000
        batch_index = 0
        with gzip.open('inputs/data_patch/data.pkl.gz', 'wb') as f:
            while batch_index < images.shape[0] - num_patches_per_batch:
                pkl.dump((
                    images[batch_index: batch_index + num_patches_per_batch],
                    labels[batch_index: batch_index + num_patches_per_batch]),
                    f)
                batch_index += num_patches_per_batch
            pkl.dump((images[batch_index:], labels[batch_index:]), f)

    def load_data(self):
        with gzip.open('inputs/data_patch/data.pkl.gz') as f:
            images, labels = pkl.load(f)
        return images, labels
        '''
