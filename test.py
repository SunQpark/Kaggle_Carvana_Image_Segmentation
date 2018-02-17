from keras.models import load_model
from utils.custom_functions import *
from utils.rle_mask import rle_encode
from model import *
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from PIL import Image


def batch_id_generator(batch_size=4):
    id_list = [id.replace('.jpg', '') for id in os.listdir('inputs/test')]
    num_id = len(id_list)
    num_loops = int(num_id/batch_size)
    iterator = 0
    for i in range(num_loops+1):
        start = batch_size * i
        end = min([batch_size * (i + 1), num_id])
        
        id_batch = id_list[start:end]
        yield id_batch

if __name__ == '__main__':
    model_name = 'Unet_180216_samepad'
    out_dir = 'outputs/'
    img_size = (1280, 1918,)
    model = load_model('models/{}.hdf5'.format(model_name), custom_objects=custom_objects())

    with open(out_dir + 'submission.csv', 'w', newline='') as csv_file:
        write_model = csv.writer(csv_file)
        write_model.writerow(["img", "rle_mask"])

        id_gen = batch_id_generator()
        while True:
            try:
                id_batch = next(id_gen)
            except(StopIteration):
                print('Done.')
                break

            images = np.empty((len(id_batch),) + img_size + (3,))
            for i, filename in enumerate(id_batch):
                img_path = "inputs/test/{}.jpg".format(filename)
                img = np.array(Image.open(img_path))
                img = img/255
                images[i, :, :, :] = img[0:1280, 0:1918, :]
            
            label_result = (model.predict(images) > 0.5)

            for i, filename in enumerate(id_batch):
                rle_mask = rle_encode(label_result[i, :, :, 0])
                write_model.writerow([filename + '.jpg', rle_mask])
