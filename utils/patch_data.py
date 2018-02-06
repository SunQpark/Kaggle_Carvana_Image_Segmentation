import os
import numpy as np
import pickle as pkl
import gzip
from utils.img2patch import rand_crop


class InputImage:
    def __init__(self, order, num_images_per_batch):
        self.images = None
        self.labels = None
        self.order = order  # the order of batch
        self.num_images_per_batch = num_images_per_batch

    def make_patch(self):
        id_list = [id.replace('.jpg', '') for id in os.listdir('inputs/train')]

        cnt = 0
        for i, id in enumerate(id_list):
            if i >= self.order * self.num_images_per_batch:
                images_patch, labels_patch = rand_crop(id)
                self.images = np.concatenate((self.images, images_patch), axis=0)
                self.labels = np.concatenate((self.labels, labels_patch), axis=0)
                cnt += 1
                if cnt == self.num_images_per_batch:
                    break

        num_data = self.images.shape[0]
        shuffle = np.random.permutation(range(num_data))
        self.images = self.images[shuffle]
        self.labels = self.labels[shuffle]

        return self.images, self.labels

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
