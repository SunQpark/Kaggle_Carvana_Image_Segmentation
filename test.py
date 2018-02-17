from keras.models import load_model
from utils.custom_functions import *
from utils.rle_mask import rle_encode
from model import *
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from PIL import Image


model_name = 'Unet_180216_samepad'
out_dir = 'outputs/'

id_list = [id.replace('.jpg', '') for id in os.listdir('inputs/test')]
total = len(id_list)


with open('test_mask.csv', 'w', newline='') as csvfile:
    write_model = csv.writer(csvfile)
    write_model.writerow(["img", "rle_mask"])

    for x, filename in enumerate(id_list):
        img_path = "inputs/test/{}.jpg".format(filename)
        img = np.array(Image.open(img_path))
        img = img/255
        img = img[np.newaxis, 0:1280, 0:1918, :]

        model = load_model('models/{}.hdf5'.format(model_name), custom_objects=custom_objects())

        label_result = model.predict(img)
        label_result = np.squeeze(label_result)

        for i in range(1280):
            for j in range(1918):
                if label_result[i, j] > 0.5:
                    label_result[i, j] = 1
                else:
                    label_result[i,j] = 0
        
        label_result = label_result.astype(np.uint8)

        write_model.writerow([filename, rle_encode(label_result)])
        print(filename, 'has completed (',x+1, 'out of', total, ')')
