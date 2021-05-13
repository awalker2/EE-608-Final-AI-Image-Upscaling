import cv2
import numpy as np
from skimage.util import random_noise
from keras import Model, Input
from keras.layers import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
import tensorflow as tf

def show_images(x_rgb, y_rgb, test_opencv_rgb, out_rgb, index):
    cv2.imshow('Image Before: ' + str(index), x_rgb[index])
    cv2.imshow('Image Desired: ' + str(index), y_rgb[index])
    cv2.imshow('Image Predicted: ' + str(index), out_rgb[index])
    cv2.imshow('OpenCV Prediction With Noise: ' + str(index), test_opencv_rgb[index])

def get_mse(x, y):
    a = x.flatten()
    b = y.flatten()
    return ((a - b)**2).mean()

def get_model_max_pooling(upscale_factor=2, channels=3):
    conv_args = {
        "activation": "relu",
        "padding": "same",
    }
    inputs = Input(shape=(None, None, channels))
    x = Conv2D(16, (3, 3), **conv_args)(inputs)
    x = Conv2D(32, (3, 3), **conv_args)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), **conv_args)(x)
    x = Conv2D(64, (3, 3), **conv_args)(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), **conv_args)(x)
    x = Conv2D(channels * (upscale_factor ** 2), 3, **conv_args)(x)
    outputs = tf.nn.depth_to_space(x, upscale_factor)

    return Model(inputs, outputs)

def get_model_base(upscale_factor=2, channels=3):
    conv_args = {
        "activation": "relu",
        "padding": "same",
    }
    inputs = Input(shape=(None, None, channels))
    x = Conv2D(16, (3, 3), **conv_args)(inputs)
    x = Conv2D(32, (3, 3), **conv_args)(x)
    x = Conv2D(64, (3, 3), **conv_args)(x)
    x = Conv2D(64, (3, 3), **conv_args)(x)
    x = Conv2D(128, (3, 3), **conv_args)(x)
    x = Conv2D(channels * (upscale_factor ** 2), 3, **conv_args)(x)
    outputs = tf.nn.depth_to_space(x, upscale_factor)

    return Model(inputs, outputs)

def get_model_simplier(upscale_factor=2, channels=3):
    conv_args = {
        "activation": "relu",
        "padding": "same",
    }
    inputs = Input(shape=(None, None, channels))
    x = Conv2D(32, (3, 3), **conv_args)(x)
    x = Conv2D(64, (3, 3), **conv_args)(x)
    x = Conv2D(128, (3, 3), **conv_args)(x)
    x = Conv2D(channels * (upscale_factor ** 2), 3, **conv_args)(x)
    outputs = tf.nn.depth_to_space(x, upscale_factor)

    return Model(inputs, outputs)

def get_model_more_complex(upscale_factor=2, channels=3):
    conv_args = {
        "activation": "relu",
        "padding": "same",
    }
    inputs = Input(shape=(None, None, channels))
    x = Conv2D(16, (3, 3), **conv_args)(inputs)
    x = Conv2D(16, (3, 3), **conv_args)(inputs)
    x = Conv2D(32, (3, 3), **conv_args)(x)
    x = Conv2D(64, (3, 3), **conv_args)(x)
    x = Conv2D(64, (3, 3), **conv_args)(x)
    x = Conv2D(128, (3, 3), **conv_args)(x)
    x = Conv2D(256, (3, 3), **conv_args)(x)
    x = Conv2D(channels * (upscale_factor ** 2), 3, **conv_args)(x)
    outputs = tf.nn.depth_to_space(x, upscale_factor)

def get_dataset(num_samples, noise_amount, orig_size, low_size, color_channels):
    # Initialize np array first, then reassign for better performance
    x = np.empty([num_samples, low_size, low_size, color_channels])
    y = np.empty([num_samples, orig_size, orig_size, color_channels])
    test_opencv = np.empty([num_samples, orig_size, orig_size, color_channels])
    for i in range(1, num_samples+1):
        img_orig = cv2.imread(f'preprocessed_images/{str(i)}.jpg')
        img_low_res = cv2.resize(img_orig, dsize=(low_size, low_size), interpolation=cv2.INTER_AREA)
        img_low_res_noise = random_noise(img_low_res, mode='s&p', amount=noise_amount)
        img_low_res_upscaled_test = cv2.resize(img_low_res_noise, dsize=(orig_size, orig_size), interpolation=cv2.INTER_AREA)

        # Adding the noise resizes the rgb values to 0-1
        x[i-1] = img_low_res_noise
        # But the orinal image is still 0-255
        y[i-1] = img_orig / 255.0
        test_opencv[i-1] = img_low_res_upscaled_test
    return x, y, test_opencv

def run_model(model, batch_size, epochs, x, y, test_opencv, open_images=False):
    # Finish model
    model.compile(optimizer='rmsprop',loss='mse')
    # Train the neural network
    model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs)
    # Process model out back to np.uint8 type - round to nearest ints
    out = model.predict(x) * 255.0
    out_rgb = out.clip(0, 255).astype(np.uint8)
    y_rgb = np.rint((y * 255.0)).astype(np.uint8)
    x_rgb = np.rint((x * 255.0)).astype(np.uint8)
    test_opencv_rgb = np.rint((test_opencv * 255.0)).astype(np.uint8)

    # Get metrics from model
    print('MSE of Model (Train): ' + str(get_mse(y_rgb, out_rgb)))
    print('MSE of OpenCV Simple Resize: ' + str(get_mse(y_rgb, test_opencv_rgb)))

    if (open_images):
        # Show Example Images
        show_images(x_rgb, y_rgb, test_opencv_rgb, 0)
        show_images(x_rgb, y_rgb, test_opencv_rgb, 100)
        show_images(x_rgb, y_rgb, test_opencv_rgb, 150)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Image params
noise_amount = 0.0005
upscale_factor = 2
orig_size = 512
low_size = 256
color_channels = 3
# Model params
num_samples = 200
batch_size = 25
epochs = 5

x, y, test_opencv = get_dataset(num_samples, noise_amount, orig_size, low_size, color_channels)
model_base = get_model_base(upscale_factor, color_channels)
run_model(model_base, batch_size, epochs, x, y, test_opencv, True)


# # Image params
# noise_amount = 0.0005
# upscale_factor = 2
# orig_size = 512
# low_size = 256
# color_channels = 3
# # Model params
# num_samples = 200
# batch_size = 25
# epochs = 100

# x, y, test_opencv = get_dataset(num_samples, noise_amount, orig_size, low_size, color_channels)
# model = get_model_base(upscale_factor, color_channels)

# # Finish model
# model.compile(optimizer='rmsprop',loss='mse')
# # Train the neural network
# model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs)
# # Process model out back to np.uint8 type
# out = model.predict(x) * 255.0
# out_rgb = out.clip(0, 255).astype(np.uint8)
# y_rgb = (y * 255.0).astype(np.uint8)
# x_rgb = (x * 255.0).astype(np.uint8)
# test_opencv_rgb = (test_opencv * 255.0).astype(np.uint8)

# # Get metrics from model
# print('MSE of Model: ' + str(get_mse(y_rgb, out_rgb)))
# print('MSE of OpenCV Simple Resize: ' + str(get_mse(y_rgb, test_opencv_rgb)))

# # Show Example Images
# show_images(x_rgb, y_rgb, test_opencv_rgb, 0)
# show_images(x_rgb, y_rgb, test_opencv_rgb, 10)
# cv2.waitKey(0)
# cv2.destroyAllWindows()