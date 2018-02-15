import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.models import load_model
from utils.custom_functions import *

def plot_save(img, label, pred):
    pass

if __name__ == '__main__':    
    model_name = 'Unet_180215_ce_with_l2'
    filename = ['0cdf5b5d0ce1_06', '0ce66b539f52_15', '0d1a9caf4350_03', '0d53224da2b7_03', 'f1eb080c7182_03']
    model_path = 'models/{}.hdf5'.format(model_name)
    model = load(file_name='Unet_180215_ce_with_l2')

    #load image and corresponding label
    index = 4

    img = np.array(Image.open("inputs/train/original/{}.jpg".format(filename[index])))
    label = np.array(Image.open("inputs/train_mask/original/{}_mask.gif".format(filename[index])))

    prediction = model.predict(img[np.newaxis, 0:1268, 0:1908, :])

    fig = plt.figure(figsize=(15, 10), dpi=200)
    fig.subplots_adjust(hspace=0.1, wspace=0.05)
    # original image
    plt.subplot(221)
    plt.axis('off')
    plt.imshow(img[0:1268, 0:1908, :])
    plt.title('Original Image')

    # true mask
    plt.subplot(222)
    plt.axis('off')
    plt.imshow(label)
    plt.title('True Label')

    # computed prediction
    plt.subplot(224)
    plt.axis('off')
    plt.imshow(prediction[0, :, :, 0])
    plt.title('Prediction')

    # computed prediction applied on image
    plt.subplot(223)
    plt.axis('off')
    plt.imshow(img[0:1268, 0:1908, :] * prediction[0, :, :, :]/255)
    plt.title('Prediction on Image')

    plt.savefig('outputs/{}_{}.png'.format(model_name, index))