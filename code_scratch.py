import cv2
import numpy as np
from skimage.util import random_noise
from keras import Model, Input
from keras.layers import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten
import tensorflow as tf

def get_dataset(num_samples, train_size, test_size, noise_amount, orig_size, low_size, color_channels):
    # Initialize np array first, then reassign for better performance
    x = np.empty([num_samples, low_size, low_size, color_channels])
    y = np.empty([num_samples, orig_size, orig_size, color_channels])
    opencv = np.empty([num_samples, orig_size, orig_size, color_channels])
    for i in range(1, num_samples+1):
        img_orig = cv2.imread(f'preprocessed_images/{str(i)}.jpg')
        img_low_res = cv2.resize(img_orig, dsize=(low_size, low_size), interpolation=cv2.INTER_AREA)
        img_low_res_noise = random_noise(img_low_res, mode='s&p', amount=noise_amount)
        img_low_res_upscaled_test = cv2.resize(img_low_res_noise, dsize=(orig_size, orig_size))

        # Adding the noise resizes the rgb values to 0-1
        x[i-1] = img_low_res_noise
        # But the orinal image is still 0-255
        y[i-1] = img_orig / 255.0
        opencv[i-1] = img_low_res_upscaled_test
    
    x_test = x[-test_size:]
    y_test = y[-test_size:]
    x_train = x[:train_size]
    y_train = y[:train_size]
    opencv_test = opencv[-test_size:]
    opencv_train = opencv[:train_size]
        
    return x_train, y_train, x_test, y_test, opencv_test, opencv_train

def show_images(x_rgb, y_rgb, out_rgb, opencv_rgb, index, test_or_train):
    cv2.imshow(test_or_train + ' Image Before: ' + str(index), x_rgb[index])
    cv2.imshow(test_or_train + ' Image Desired: ' + str(index), y_rgb[index])
    cv2.imshow(test_or_train + ' Image Predicted: ' + str(index), out_rgb[index])
    cv2.imshow(test_or_train + ' OpenCV Prediction With Noise: ' + str(index), opencv_rgb[index])

def get_mse(x, y):
    a = x.flatten()
    b = y.flatten()
    return (np.square(a - b).mean())

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

def run_model(model, batch_size, epochs, optimizer, x_train, y_train, x_test, y_test, opencv_test, opencv_train):
    # Finish model
    model.compile(optimizer=optimizer, loss='mse')
    # Train the neural network
    model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs)
    
    # Process model out back to np.uint8 type - round to nearest ints
    out_train = model.predict(x_train) * 255.0
    out_rgb_train = out_train.clip(0, 255).astype(np.uint8)
    y_rgb_train = np.rint((y_train * 255.0)).astype(np.uint8)
    x_rgb_train = np.rint((x_train * 255.0)).astype(np.uint8)

    out_test = model.predict(x_test) * 255.0
    out_rgb_test = out_test.clip(0, 255).astype(np.uint8)
    y_rgb_test = np.rint((y_test * 255.0)).astype(np.uint8)
    x_rgb_test = np.rint((x_test * 255.0)).astype(np.uint8)
    
    opencv_rgb_train = np.rint((opencv_train * 255.0)).astype(np.uint8)
    opencv_rgb_test = np.rint((opencv_test * 255.0)).astype(np.uint8)

    # Get metrics from model
    print('MSE of Model (Train): ' + str(get_mse(y_rgb_train, out_rgb_train)))
    print('MSE of Model (Test): ' + str(get_mse(y_rgb_test, out_rgb_test)))
    # print('MSE of OpenCV Simple Resize: ' + str(get_mse(y_rgb, test_opencv_rgb)))

    # Show Example Images
    show_images(x_rgb_train, y_rgb_train, out_rgb_train, opencv_rgb_train, 0, 'Training')
    show_images(x_rgb_test, y_rgb_test, out_rgb_test, opencv_rgb_test, 20, 'Testing')
    show_images(x_rgb_test, y_rgb_test, out_rgb_test, opencv_rgb_test, 39, 'Testing')
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
batch_size = 20
train_size = 160
test_size = 40
epochs = 5
optimzer = 'rmsprop'

x_train, y_train, x_test, y_test, opencv_test, opencv_train = get_dataset(num_samples, train_size, test_size, noise_amount, orig_size, low_size, color_channels)
model_base = get_model_base(upscale_factor, color_channels)
run_model(model_base, batch_size, epochs, optimzer, x_train, y_train, x_test, y_test, opencv_test, opencv_train)