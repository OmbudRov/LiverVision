import tensorflow.keras.backend as K

def DiceCoef(y_true, y_pred, smooth=1.e-9):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)

def DiceCoefLoss(y_true, y_pred):
    return 1 - DiceCoef(y_true, y_pred)