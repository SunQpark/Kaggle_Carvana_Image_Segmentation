import os
import numpy as np
from PIL import Image
from model_keras import model

if __name__ == '__main__':
    file_list = [filename.replace('.jpg', '') for filename in os.listdir('inputs/train/')] 

    images = []
    labels = []

    for filename in file_list:
        img = np.array(Image.open("inputs/train/{}.jpg".format(filename)))
        mask = np.array(Image.open("inputs/train_masks/{}_mask.gif".format(filename)))
        
        images.append(img)
        labels.append(mask)

    images = np.array(images)/255
    labels = np.array(labels)[:, :, :, np.newaxis]
    labels = np.pad(labels, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant', constant_values=0)
    print(labels.shape)

    model = model(images, labels)

    model.fit(images, labels)