import numpy as np  
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
# from utils.rle_mask import rle_decode

def set_data_gen():
    # we create two instances with the same arguments
    data_gen_args = dict(rotation_range=90.,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=0.2,
                        rescale=1./255)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # Provide the same seed and keyword arguments to the flow methods
    seed = 1

    image_generator = image_datagen.flow_from_directory(
        'inputs/train', 
        target_size=(1268, 1908),
        class_mode=None,
        color_mode='rgb',
        batch_size=2,
        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
        'inputs/train_mask', 
        target_size=(1268, 1908),
        class_mode=None,
        color_mode='grayscale',
        batch_size=2,
        seed=seed
    )

    # combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator)
    return train_generator


if __name__ == '__main__':
    train_gen = set_data_gen()
    data, = train_gen.yields

