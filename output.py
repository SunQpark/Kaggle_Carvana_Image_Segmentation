from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

filename = '0cdf5b5d0ce1_01'

img = np.array(Image.open("inputs/train/train/original/{}.jpg".format(filename)))
label = np.array(Image.open("inputs/train_mask/train_mask/original/{}_mask.gif".format(filename)))

plt.imshow(img)
plt.show()

plt.imshow(label)
plt.show()

model = load_model('models/model_180211.hdf5')
img = img[np.newaxis, 0:1268, 0:1908, :]
label_result = model.predict(img)
label_result = np.squeeze(label_result)
np.savetxt("hi.csv", label_result, delimiter=",")

print(label_result.shape)
plt.imshow(label_result)
plt.show()
