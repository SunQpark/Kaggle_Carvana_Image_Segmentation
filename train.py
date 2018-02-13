import os
import numpy as np
from PIL import Image
from keras.models import load_model
from utils.data_generator import set_data_gen
from model import Unet
from losses import custom_objects

if __name__ == '__main__':
    train_generator = set_data_gen()

    model_name = 'Unet_180212'
    model_path = 'models/{}.hdf5'.format(model_name)

    batch_size = 3
    num_loops = 2
    epochs_per_loop = 3
    steps_per_epoch = int(5088/batch_size)

    if os.path.isfile(model_path) is True:
        model = load_model(model_path, custom_objects=custom_objects())
    else:
        model = Unet()

    for loop in range(num_loops):
        model.fit_generator(train_generator, 
        steps_per_epoch=steps_per_epoch,
        epochs=epochs_per_loop,
        verbose=1)

        model.save(model_path)
