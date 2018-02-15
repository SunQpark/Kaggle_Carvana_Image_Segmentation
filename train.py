import os
import numpy as np
from PIL import Image
from keras.models import load_model
from datetime import datetime

from model import Unet
from utils.data_generator import set_data_gen
from utils.custom_functions import *

if __name__ == '__main__':
    # formatting path to the model file to be saved/loaded
    model_path = model_path('Unet', 'ce_with_l2')
    # model_name = 'Unet'
    # now = datetime.now()
    # today = str(now.year)[-2:] + str(now.month).zfill(2) + str(now.day).zfill(2)
    # loss_name = 'l2'
    # model_path = 'models/{}_{}_{}.hdf5'.format(model_name, today, loss_name)
    # model_path = 'models/Unet_180215_l2.hdf5'.format(model_name, today, loss)

    train_generator = set_data_gen()
    
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
