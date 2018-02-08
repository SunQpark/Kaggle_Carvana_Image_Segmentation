from keras.models import Model, load_model ,Sequential
from keras.layers import Input, concatenate, Cropping2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Conv2DTranspose
from keras.utils import np_utils


def model(X_input, y):
    X_input = Input(shape=(None, None, 3))
    X = Conv2D(64, (3, 3), activation='relu')(X_input)
    X = Conv2D(64, (3, 3), activation='relu')(X)
    X0 = Cropping2D(cropping=((4, 4), (4, 4)))(X)

    X = MaxPool2D(pool_size=(2, 2))(X)
    X = Conv2D(128, (3, 3), activation='relu')(X)
    X = Conv2D(128, (3, 3), activation='relu')(X)
    X1 = Cropping2D(cropping=((4, 4), (4, 4)))(X)

    X = MaxPool2D(pool_size=(2, 2))(X)
    X = Conv2D(256, (3, 3), activation='relu')(X)
    X = Conv2D(256, (3, 3), activation='relu')(X)
    X2 = Cropping2D(cropping=((4, 4), (4, 4)))(X)

    X = MaxPool2D(pool_size=(2, 2))(X)
    X = Conv2D(512, (3, 3), activation='relu')(X)
    X = Conv2D(512, (3, 3), activation='relu')(X)
    X3 = Cropping2D(cropping=((4, 4), (4, 4)))(X)

    X = MaxPool2D(pool_size=(2, 2))(X)
    X = Conv2D(1024, (3, 3), activation='relu')(X)
    X = Conv2D(1024, (3, 3), activation='relu')(X)

    X = Conv2DTranspose(512, (2, 2))(X)
    X = concatenate([X3, X])
    X = Conv2D(512, (3, 3), activation='relu')(X)
    X = Conv2D(512, (3, 3), activation='relu')(X)

    X = Conv2DTranspose(256, (2, 2))(X)
    X = concatenate([X2, X])
    X = Conv2D(256, (3, 3), activation='relu')(X)
    X = Conv2D(256, (3, 3), activation='relu')(X)

    X = Conv2DTranspose(128, (2, 2))(X)
    X = concatenate([X1, X])
    X = Conv2D(128, (3, 3), activation='relu')(X)
    X = Conv2D(128, (3, 3), activation='relu')(X)
        
    X = Conv2DTranspose(64, (2, 2))(X)
    X = concatenate([X0, X])
    X = Conv2D(64, (3, 3), activation='relu')(X)
    X = Conv2D(64, (3, 3), activation='relu')(X)
    
    X_out = Conv2D(1, (1, 1), activation='sigmoid')(X)

    model = Model(input=X_input, outputs=X_out)

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    # model.fit(X_train, y_train,
    #         batch_size=32, epochs=10, verbose=1)

    # score = model.evaluate(X_test, y_test, verbose=1)

    # model.save("models/model.hdf5")

if __name__ == '__main__':
    pass