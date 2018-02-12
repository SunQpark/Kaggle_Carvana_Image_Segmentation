import keras.backend as K
from keras.losses import mean_squared_error

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)

def dice_with_l2_loss(y_true, y_pred):
    l2_loss = mean_squared_error(y_true, y_pred)
    dice_loss = 1 - dice_coef(y_true, y_pred)
    return l2_loss + dice_loss