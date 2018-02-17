from keras.models import load_model
from utils.custom_functions import *
from model import *
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image



filename = '0ce66b539f52_11'
model_name = 'Unet_180216_samepad'
out_dir = 'outputs/'

img_path = "inputs/train/original/{}.jpg".format(filename)
mask_path = "inputs/train_mask/original/{}_mask.gif".format(filename)

img = np.array(Image.open(img_path))
img = img/255
label_true = np.array(Image.open(mask_path))
label_true = label_true/255
y_true = label_true[np.newaxis, :, :, np.newaxis]
print(y_true.shape)

plt.imshow(label_true)
plt.show()

model = load_model('models/{}.hdf5'.format(model_name), custom_objects=custom_objects())

img = img[np.newaxis, 0:1280, 0:1918, :]
label_result = model.predict(img)
y_pred = label_result
label_result = np.squeeze(label_result)

#print(dice_coef(y_true=y_true, y_pred=y_pred))

# np.savetxt("hi.csv", label_result, delimiter=",")
label_pred = Image.fromarray(label_result)
# label_pred.save(out_dir + filename + '_out.png')
plt.imshow(label_result)
plt.show()
