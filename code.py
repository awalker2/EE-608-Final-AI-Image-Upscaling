import cv2
import numpy as np
from skimage.util import random_noise
from keras import Model, Input
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation, InputLayer, UpSampling2D, UpSampling3D
from keras.layers.convolutional import Conv2D, Conv3D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from skimage.io import imsave
import tensorflow as tf

def get_model(upscale_factor=2, channels=3):
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }
    inputs = Input(shape=(None, None, channels))
    x = Conv2D(64, 5, **conv_args)(inputs)
    x = Conv2D(64, 3, **conv_args)(x)
    x = Conv2D(32, 3, **conv_args)(x)
    x = Conv2D(channels * (upscale_factor ** 2), 3, **conv_args)(x)
    outputs = tf.nn.depth_to_space(x, upscale_factor)

    return Model(inputs, outputs)

# Desired Image
img = cv2.imread('img.jpg')
high_res = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_AREA)
# Input
low_res = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_AREA)
low_res_noise = random_noise(low_res, mode='s&p', amount=0.01)

X = low_res_noise.reshape(1, 256, 256, 3)
Y = high_res.reshape(1, 512, 512, 3) / 255.0

model = get_model(2, 3)

# Finish model
model.compile(optimizer='rmsprop',loss='mse')

#Train the neural network
model.fit(x=X, y=Y, batch_size=1, epochs=1000)

print(model.evaluate(X, Y, batch_size=1))

out = model.predict(X)[0] * 255.0
prediction = out.clip(0, 255).astype(np.uint8)

cv2.imshow('image before', low_res_noise)
cv2.imshow('image desired', high_res)
cv2.imshow('image predicted', prediction)
cv2.waitKey(0)
cv2.destroyAllWindows()


