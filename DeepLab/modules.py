from keras.models import Model
from keras.layers import Conv2D


def aspp(X, depth):
    X = Conv2D(filters=depth, kernel_size=(3, 3), dilation_rate=(6, 6), activation='relu')(X)
    X = Conv2D(filters=depth, kernel_size=(3, 3), dilation_rate=(12, 12), activation='relu')(X)
    X = Conv2D(filters=depth, kernel_size=(3, 3), dilation_rate=(18, 18), activation='relu')(X)
    X = Conv2D(filters=depth, kernel_size=(3, 3), dilation_rate=(24, 24), activation='relu')(X)
    return X
    