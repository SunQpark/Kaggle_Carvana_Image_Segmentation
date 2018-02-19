import keras.backend as K
from keras.models import load_model
from keras.losses import mean_squared_error, binary_crossentropy
from datetime import datetime

def load(model_name='Unet', loss_name='l2', file_name=None):
    if file_name is None:
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
        'crossentropy_with_l2':crossentropy_with_l2,
        'bde_dice_loss':bde_dice_loss
        }


# dice coef, loss from https://github.com/SunQpark/Kaggle-Carvana-3rd-Place-Solution/blob/master/losses.py
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def dice_with_l2_loss(y_true, y_pred, weight_l2=1.0):
    l2_loss = mean_squared_error(y_true, y_pred)
    dice_loss = dice_coef(y_true, y_pred)
    total_loss = weight_l2 * l2_loss + (1.0 - weight_l2)*dice_loss
    return total_loss

def crossentropy_with_l2(y_true, y_pred, weight_l2=2.0):
    crossentropy = binary_crossentropy(y_true, y_pred)
    l2 = mean_squared_error(y_true, y_pred)
    return crossentropy + l2 * weight_l2

if __name__ == '__main__':
    model = load()
    model.summary()