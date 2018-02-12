from keras.models import load_model
from model_keras import dice_coef, dice_with_l2_loss
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

filename = '0cdf5b5d0ce1_05'

img = np.array(Image.open("inputs/train/original/{}.jpg".format(filename)))
# label_orig = np.array(Image.open("inputs/train_mask/original/{}_mask.gif".format(filename)))

model = load_model('models/model_180212.hdf5', 
    custom_objects={'dice_with_l2_loss':dice_with_l2_loss, 'dice_coef':dice_coef})
    
img = img[np.newaxis, 0:1268, 0:1908, :]
label_result = model.predict(img)
label_result = np.squeeze(label_result)
# np.savetxt("hi.csv", label_result, delimiter=",")

print(label_result.shape)
plt.imshow(label_result)
plt.show()
