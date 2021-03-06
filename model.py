from keras.models import Model, load_model ,Sequential
from keras.layers import Input, concatenate, Cropping2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D, ZeroPadding2D
from keras.layers import Conv2DTranspose
from keras.utils import np_utils
from keras import backend as K
from utils.custom_functions import *

def Unet():
    conv_kwarg = dict(padding='same', activation='elu', kernel_initializer='he_normal')

    X_input = Input(shape=(None, None, 3))
    X = ZeroPadding2D(padding=(0, 1))(X_input)
    X = Conv2D(16, (3, 3), **conv_kwarg)(X)
    X0 = Conv2D(16, (3, 3), **conv_kwarg)(X)
    # X0 = Cropping2D(cropping=((88, 88), (88, 88)))(X)

    X = MaxPool2D(pool_size=(2, 2), padding='same')(X0)
    X = Conv2D(32, (3, 3), **conv_kwarg)(X)
    X1 = Conv2D(32, (3, 3), **conv_kwarg)(X)
    # X1 = Cropping2D(cropping=((40, 40), (40, 40)))(X)

    X = MaxPool2D(pool_size=(2, 2), padding='same')(X1)
    X = Conv2D(64, (3, 3), **conv_kwarg)(X)
    X2 = Conv2D(64, (3, 3), **conv_kwarg)(X)
    # X2 = Cropping2D(cropping=((16, 16), (16, 16)))(X)

    X = MaxPool2D(pool_size=(2, 2), padding='same')(X2)
    X = Conv2D(128, (3, 3), **conv_kwarg)(X)
    X3 = Conv2D(128, (3, 3), **conv_kwarg)(X)
    # X3 = Cropping2D(cropping=((4, 4), (4, 4)))(X)

    X = MaxPool2D(pool_size=(2, 2), padding='same')(X3)
    X = Conv2D(256, (3, 3), **conv_kwarg)(X)
    X = Conv2D(256, (3, 3), dilation_rate=2, **conv_kwarg)(X)
    X = Conv2D(256, (3, 3), dilation_rate=4, **conv_kwarg)(X)
    X = Conv2D(256, (3, 3), dilation_rate=8, **conv_kwarg)(X)
    X = Conv2D(256, (3, 3), dilation_rate=16, **conv_kwarg)(X)
    X = Conv2D(256, (3, 3), dilation_rate=32, **conv_kwarg)(X)


    X = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(X)
    X = concatenate([X3, X])
    X = Conv2D(128, (3, 3), **conv_kwarg)(X)
    X = Conv2D(128, (3, 3), **conv_kwarg)(X)

    X = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(X)
    X = concatenate([X2, X])
    X = Conv2D(64, (3, 3), **conv_kwarg)(X)
    X = Conv2D(64, (3, 3), **conv_kwarg)(X)

    X = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(X)
    X = concatenate([X1, X])
    X = Conv2D(32, (3, 3), **conv_kwarg)(X)
    X = Conv2D(32, (3, 3), **conv_kwarg)(X)
        
    X = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(X)
    X = concatenate([X0, X])
    X = Conv2D(16, (3, 3), **conv_kwarg)(X)
    X = Conv2D(16, (3, 3), **conv_kwarg)(X)
    
    X_out = Conv2D(1, (1, 1), activation='sigmoid')(X)
    X_out = Cropping2D(cropping=((0, 0), (1, 1)))(X_out)

    model = Model(inputs=X_input, outputs=X_out)

    model.compile(loss=bce_dice_loss,
                optimizer='adam',
                metrics=[dice_coef])
    return model

if __name__ == '__main__':
    model = Unet()
    model.summary()
