import os
import numpy as np
from PIL import Image
from utils.data_generator import set_data_gen
from model_keras import Unet

if __name__ == '__main__':

    train_generator = set_data_gen()
    model = Unet()
    
    # model.fit(images, labels, epochs=10)
    model.fit_generator(train_generator, 
    steps_per_epoch=2544,
    epochs=5,
    verbose=1)

    model.save('models/model_180210.hdf5')
