import keras.backend as K
from keras.models import load_model
from keras.losses import mean_squared_error, binary_crossentropy
from datetime import datetime

def load(filename=None, model_name='Unet', loss_name='l2'):
    if filename is None:
        path = model_path(model_name, loss_name)
    else:
        path = 'models/{}.hdf5'.format(file_name)
    return load_model(path, custom_objects=custom_objects())

def model_path(model_name, loss_name):
    now = datetime.now()
    today = str(now.year)[-2:] + str(now.month).zfill(2) + str(now.day).zfill(2)
    path = 'models/{}_{}_{}.hdf5'.format(model_name, today, loss_name)
    return path

def custom_objects():
    return {
        'dice_coef':dice_coef, 
        'dice_with_l2_loss':dice_with_l2_loss, 
        'crossentropy_with_l2':crossentropy_with_l2
        }

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)

def dice_with_l2_loss(y_true, y_pred, weight_l2=1.0):
    l2_loss = mean_squared_error(y_true, y_pred)
    dice_loss = 1 - dice_coef(y_true, y_pred)
    total_loss = weight_l2 * l2_loss + (1.0 - weight_l2)*dice_loss
    return total_loss

def crossentropy_with_l2(y_true, y_pred, weight_l2=10.0):
    crossentropy = binary_crossentropy(y_true, y_pred)
    l2 = mean_squared_error(y_true, y_pred)
    return crossentropy + l2 * weight_l2

if __name__ == '__main__':
    model = load()
    model.summary()