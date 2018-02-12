import keras.backend as K
from keras.models import Model, load_model ,Sequential
from keras.layers import Input, concatenate, Cropping2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D, ZeroPadding2D
from keras.layers import Conv2DTranspose
from keras.losses import mean_squared_error
from keras.utils import np_utils


def Unet():
    X_input = Input(shape=(None, None, 3))
    X = ZeroPadding2D(padding=(92, 92))(X_input)
    X = Conv2D(16, (3, 3), activation='relu')(X)
    X = Conv2D(16, (3, 3), activation='relu')(X)
    X0 = Cropping2D(cropping=((88, 88), (88, 88)))(X)

    X = MaxPool2D(pool_size=(2, 2), padding='same')(X)
    X = Conv2D(32, (3, 3), activation='relu')(X)
    X = Conv2D(32, (3, 3), activation='relu')(X)
    X1 = Cropping2D(cropping=((40, 40), (40, 40)))(X)

    X = MaxPool2D(pool_size=(2, 2), padding='same')(X)
    X = Conv2D(64, (3, 3), activation='relu')(X)
    X = Conv2D(64, (3, 3), activation='relu')(X)
    X2 = Cropping2D(cropping=((16, 16), (16, 16)))(X)

    X = MaxPool2D(pool_size=(2, 2), padding='same')(X)
    X = Conv2D(128, (3, 3), activation='relu')(X)
    X = Conv2D(128, (3, 3), activation='relu')(X)
    X3 = Cropping2D(cropping=((4, 4), (4, 4)))(X)

    X = MaxPool2D(pool_size=(2, 2), padding='same')(X)
    X = Conv2D(256, (3, 3), activation='relu')(X)
    X = Conv2D(256, (3, 3), activation='relu')(X)

    X = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(X)
    X = concatenate([X3, X])
    X = Conv2D(128, (3, 3), activation='relu')(X)
    X = Conv2D(128, (3, 3), activation='relu')(X)

    X = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(X)
    X = concatenate([X2, X])
    X = Conv2D(64, (3, 3), activation='relu')(X)
    X = Conv2D(64, (3, 3), activation='relu')(X)

    X = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(X)
    X = concatenate([X1, X])
    X = Conv2D(32, (3, 3), activation='relu')(X)
    X = Conv2D(32, (3, 3), activation='relu')(X)
        
    X = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(X)
    X = concatenate([X0, X])
    X = Conv2D(16, (3, 3), activation='relu')(X)
    X = Conv2D(16, (3, 3), activation='relu')(X)
    
    X_out = Conv2D(1, (1, 1), activation='sigmoid')(X)

    model = Model(inputs=X_input, outputs=X_out)

    model.compile(loss=dice_with_l2_loss,
                optimizer='adam',
                metrics=[dice_coef])
    return model

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_with_l2_loss(y_true, y_pred):
    l2_loss = mean_squared_error(y_true, y_pred)
    dice_loss = 1 - dice_coef(y_true, y_pred)
    return l2_loss + dice_loss

def naive_keras(X_input, y):
    model = Sequential()
    model.add( Conv2D(16, 3, activation='relu', padding='same', input_shape=(320, 480, 12) ) )
    model.add( Conv2D(32, 3, activation='relu', padding='same') )
    model.add( Conv2D(1, 5, activation='sigmoid', padding='same') )

if __name__ == '__main__':
    pass
