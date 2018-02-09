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
    steps_per_epoch=2000,
    epochs=10,
    verbose=1)
    model.save('models/model_180209.hdf5')
    
    # file_list = [filename.replace('.jpg', '') for filename in os.listdir('inputs/train/')] 

    # images = []
    # labels = []

    # for filename in file_list:
    #     img = np.array(Image.open("inputs/train/{}.jpg".format(filename)))
    #     mask = np.array(Image.open("inputs/train_masks/{}_mask.gif".format(filename)))
        
    #     images.append(img)
    #     labels.append(mask)

    # images = np.array(images)/255
    # labels = np.array(labels)[:, :, :, np.newaxis]
    # labels = np.pad(labels, ((0, 0), (2, 2), (3, 3), (0, 0)), mode='constant', constant_values=0)
    # print(labels.shape)