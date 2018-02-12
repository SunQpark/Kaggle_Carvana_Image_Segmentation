from keras.models import load_model
from utils.losses import *
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

filename = '0cdf5b5d0ce1_05'
model_name = 'model_180212'
out_dir = 'outputs/'

img = np.array(Image.open("inputs/train/original/{}.jpg".format(filename)))
label_true = np.array(Image.open("inputs/train_mask/original/{}_mask.gif".format(filename)))

model = load_model('models/{}.hdf5'.format(model_name), 
    custom_objects={'dice_with_l2_loss':dice_with_l2_loss, 'dice_coef':dice_coef})
    
img = img[np.newaxis, 0:1268, 0:1908, :]
label_result = model.predict(img)
label_result = np.squeeze(label_result)
# np.savetxt("hi.csv", label_result, delimiter=",")
label_pred = Image.fromarray(label_result)
label_pred.save(out_dir + filename + '_out.png')
# print(label_pred.shape)
# plt.imshow(label_pred)
# plt.show()
