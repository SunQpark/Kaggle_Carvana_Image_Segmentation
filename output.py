from keras.models import load_model
from utils.custom_functions import *
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

filename = '0cdf5b5d0ce1_05'
model_name = 'Unet_180215'
out_dir = 'outputs/'

img_path = "inputs/train/original/{}.jpg".format(filename)
mask_path = "inputs/train_mask/original/{}_mask.gif".format(filename)

img = np.array(Image.open(img_path))
label_true = np.array(Image.open(mask_path))

model = load_model('models/{}.hdf5'.format(model_name), custom_objects=custom_objects())
    
img = img[np.newaxis, 0:1268, 0:1908, :]
label_result = model.predict(img)
label_result = np.squeeze(label_result)
# np.savetxt("hi.csv", label_result, delimiter=",")
label_pred = Image.fromarray(label_result)
# label_pred.save(out_dir + filename + '_out.png')
plt.imshow(label_pred)
plt.show()
