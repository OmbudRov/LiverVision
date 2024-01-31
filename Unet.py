import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, UpSampling2D

def Unet(InputShape, OutputChannels):
    filters = [64, 128, 256, 512, 1024]
    InputLayer = Input(shape=InputShape, name="InputLayer")

    # Encoder
    E1 = Conv2D(filters[0], 3, activation='relu', padding='same')(InputLayer)
    E1 = Conv2D(filters[0], 3, activation='relu', padding='same')(E1)
    E2 = MaxPooling2D(pool_size=(2, 2))(E1)
    E2 = Conv2D(filters[1], 3, activation='relu', padding='same')(E2)
    E2 = Conv2D(filters[1], 3, activation='relu', padding='same')(E2)
    E3 = MaxPooling2D(pool_size=(2, 2))(E2)
    E3 = Conv2D(filters[2], 3, activation='relu', padding='same')(E3)
    E3 = Conv2D(filters[2], 3, activation='relu', padding='same')(E3)
    E4 = MaxPooling2D(pool_size=(2, 2))(E3)
    E4 = Conv2D(filters[3], 3, activation='relu', padding='same')(E4)
    E4 = Conv2D(filters[3], 3, activation='relu', padding='same')(E4)
    E5 = MaxPooling2D(pool_size=(2, 2))(E4)
    E5 = Conv2D(filters[4], 3, activation='relu', padding='same')(E5)
    E5 = Conv2D(filters[4], 3, activation='relu', padding='same')(E5)

    # Decoder
    D4 = UpSampling2D(size=(2, 2))(E5)
    D4 = concatenate([E4, D4], axis=-1)
    D4 = Conv2D(filters[3], 3, activation='relu', padding='same')(D4)
    D4 = Conv2D(filters[3], 3, activation='relu', padding='same')(D4)
    D3 = UpSampling2D(size=(2, 2))(D4)
    D3 = concatenate([E3, D3], axis=-1)
    D3 = Conv2D(filters[2], 3, activation='relu', padding='same')(D3)
    D3 = Conv2D(filters[2], 3, activation='relu', padding='same')(D3)
    D2 = UpSampling2D(size=(2, 2))(D3)
    D2 = concatenate([E2, D2], axis=-1)
    D2 = Conv2D(filters[1], 3, activation='relu', padding='same')(D2)
    D2 = Conv2D(filters[1], 3, activation='relu', padding='same')(D2)
    D1 = UpSampling2D(size=(2, 2))(D2)
    D1 = concatenate([E1, D1], axis=-1)
    D1 = Conv2D(filters[0], 3, activation='relu', padding='same')(D1)
    D1 = Conv2D(filters[0], 3, activation='relu', padding='same')(D1)

    Output = Conv2D(OutputChannels, 1, activation='softmax')(D1)

    model = tf.keras.Model(inputs=InputLayer, outputs=Output, name="ComplexUnet")
    return model

def PrepareModel():
    InputShape = (160, 160, 3)
    OutputChannels = 1
    return Unet(InputShape, OutputChannels)
