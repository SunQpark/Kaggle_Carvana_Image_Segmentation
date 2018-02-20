import os
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from datetime import datetime

from model import Unet
from utils.data_generator import set_data_gen
from utils.custom_functions import *

if __name__ == '__main__':
    model_name = 'Unet_180220_elu'
    model_path = 'models/{}.hdf5'.format(model_name)

    batch_size = 2
    num_loops = 5
    epochs_per_loop = 5
    steps_per_epoch = int(5088/batch_size)

    train_generator = set_data_gen(batch_size)
    checkpointer = ModelCheckpoint('models/{}.h5'.format(model_name), verbose=1, save_best_only=True)

    if os.path.isfile(model_path) is True:
        print('loading model from : {}'.format(model_path))
        model = load_model(model_path, custom_objects=custom_objects())
    else:
        model = Unet()

    for loop in range(num_loops):
        print('\nTraining {}/{} \n'.format(loop + 1, num_loops))
        results = model.fit_generator(train_generator, 
            steps_per_epoch=steps_per_epoch,
            epochs=epochs_per_loop,
            callbacks=[checkpointer])

        model.save(model_path)
