import numpy as np  
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from utils.rle_mask import rle_decode

def set_data_gen():
    # we create two instances with the same arguments
    data_gen_args = dict(featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=90.,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=0.2)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    # image_datagen.fit(images, augment=True, seed=seed)
    # mask_datagen.fit(masks, augment=True, seed=seed)

    image_generator = image_datagen.flow_from_directory(
        'inputs/train', 
        target_size=(1280, 1918),
        class_mode=None,
        batch_size=2,
        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
        'inputs/train_mask', 
        target_size=(1280, 1918),
        class_mode=None,
        color_mode='grayscale',
        batch_size=2,
        seed=seed
    )
    '''
    data2 = mask_generator.next()
    dataarray = np.asarray(data2)
    np.savetxt("temp.csv", dataarray[0, :, :, 0], delimiter=",")
    '''

    '''
    # reading rle-formatted masks from csv
    rle_masks = pd.read_csv('inputs/train_masks.csv')['rle_mask']

    # decode mask data and feed it to mask data generator
    mask_shape = (1280, 1918)
    masks = np.empty((rle_masks.shape[0],)+mask_shape+(1,), dtype=np.bool)
    print(rle_masks.shape[0])
    
    for (rle_mask, mask) in zip(rle_masks, masks):
        mask = rle_decode(rle_mask, mask_shape)

    mask_generator = mask_datagen.flow(
        masks,
        batch_size=16,
        seed=seed)
    '''

    # combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator)
    return image_generator


if __name__ == '__main__':
    train_gen = set_data_gen()
    data = train_gen.yields

