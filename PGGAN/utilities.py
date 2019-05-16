'''
File    : utilities.py
Author  : Jay Santokhi, Percy Jacks and Adrian Wilczynski (jks1g15, pj2g15, aw11g15)@soton.ac.uk
Brief   : Defines utility functions for use in training PGGAN
Note    : Inspiration from:
          'GANs in action, Chapter 6: PGGAN
          https://github.com/GANs-in-Action/gans-in-action/tree/master/chapter-6
          and
          Mylittlerapture -- https://github.com/Mylittlerapture/GANLib
'''
from imutils import build_montages

import tensorflow as tf
import keras as K
import numpy as np
import cv2


def augment(data):
    '''
    Enlarging dataset by shifting 1 pixel in each direction
    '''
    off_x_p = data.copy()
    off_x_p[:, 1:, :, :] = off_x_p[:, :-1, :, :]
    off_x_m = data.copy()
    off_x_p[:, :-1, :, :] = off_x_p[:, 1:, :, :]

    off_y_p = data.copy()
    off_y_p[:, :, 1:, :] = off_y_p[:, :, :-1, :]
    off_y_m = data.copy()
    off_y_m[:, :, :-1, :] = off_y_m[:, :, 1:, :]
    data = np.concatenate((data, off_y_p, off_y_m, data, off_x_p, off_x_m), axis=0)
    return data

def equalise_learning_rate(shape, dtype, partition_info):
    '''
    This adjusts the weights of every layer by the constant from He's initializer
    so that we adjust for the variance in the dynamic range in different features

    Arguments:
        shape - shape of tensor (layer): [kernel, kernel, height, feature_maps]

    Notes: fan_in - adjustment for the number of incoming connections as per
                    Xavier's/He's initialisation
    '''
    # This gives us the number of incoming connections per neuron
    if len(shape) == 2:
        fan_in = shape[0]
    else:
        field = np.prod(shape[:-2])
        fan_in = shape[-2] * field

    # This used He's initialisation constant (He et al, 2015)
    std = np.sqrt(2/max(1., fan_in))

    adjusted_weights = tf.get_variable('layer', shape=shape,
        initializer = tf.initializers.random_normal(0,1), dtype=np.float32) * std
    return adjusted_weights

initialization = equalise_learning_rate

def new_sheet(layer, filters, kernel_size, padding, name, pix_norm=True):
    '''
    To grow layers
    '''
    layer = tf.layers.conv2d(layer, filters, kernel_size, padding=padding, name=name,
                             kernel_initializer=initialization)
    layer = tf.nn.leaky_relu(layer, alpha=0.2)
    if pix_norm: layer = pixelwise_feat_norm(layer)
    return layer

def transition_alpha(gan):
    '''
    Threshold value alpha used to smoothly merge (last) layer.
    '''
    epoch = tf.cast(gan.epoch, tf.float32)
    epochs = tf.cast(gan.epochs, tf.float32)
    a = epoch / (epochs/ 2)
    b = 1
    return tf.minimum(a, b)

def upscale_layer(layer, factor=2):
    '''
    Upscales layer (tensor) by the factor (int) where
    the tensor is [group, height, width, channels]
    '''
    height = layer.get_shape()[1]
    width = layer.get_shape()[2]
    height_factor = factor * height
    width_factor = factor * width
    size = (height_factor,width_factor)
    upscaled_layer = tf.image.resize_nearest_neighbor(layer, size)
    return upscaled_layer

def minibatch_std_layer(layer, group_size=4):
    '''
    Will calculate minibatch standard deviation for a layer.
    Will do so under a pre-specified tf-scope with Keras.
    Needs layer to be a float32 data type.
    '''
    # Minibatch group must be divisible by (or <=) group_size
    group_size = tf.minimum(group_size, tf.shape(layer)[0])

    # Just getting some shape information so that we can use
    # them as shorthand as well as to ensure defaults
    s = layer.shape

    # Reshaping so that we operate on the level of the minibatch
    # in this code we assume the layer to be:
    # [Group (G), Minibatch (M), Width (W), Height (H), Channel (C)]
    # but be careful different implementations use the Theano specific
    # order instead
    minibatch = tf.reshape(layer, [group_size, -1, s[1], s[2], s[3]])
    # MIGHT NEED TO CAST HERE
    minibatch = tf.cast(minibatch, tf.float32)
    # Center the mean over the group [M, W, H, C]
    minibatch -= tf.reduce_mean(minibatch, axis=0, keepdims=True)
    # Calculate the variance of the group [M, W, H, C]
    minibatch = tf.reduce_mean(tf.square(minibatch), axis=0)
    # Calculate the standard deviation over the group [M,W,H,C]
    minibatch = tf.square(minibatch + 1e-8)
    # Take the average over feature maps and pixels [M,1,1,1]
    minibatch = tf.reduce_mean(minibatch, axis=[1,2,3], keepdims=True)
    # Add as a layer for each group and pixels
    minibatch = tf.tile(minibatch, [group_size, 1, s[2], s[3]])
    # Append as a new feature map
    return tf.concat([layer, minibatch], axis=1)

def pixelwise_feat_norm(inputs, **kwargs):
    '''
    Uses pixelwise feature normalization as proposed by
    Krizhevsky et al. 2012. Returns the input normalized
    Arguments:
        inputs - TF Layers
    '''
    normalization_constant = K.backend.sqrt(K.backend.mean(
        inputs**2, axis=-1, keepdims=True) + 1.0e-8)
    return inputs / normalization_constant

def sample_images(gen, row, col, noise_dim, file, counter, shape):
    '''
    Creating a row by col image of the generated images
    '''
    sample_noise = np.random.uniform(-1, 1, (row * col, noise_dim))
    gen_imgs = gen.predict(sample_noise, moving_average = True)

    count = int(counter/100)
    images = gen_imgs
    images = ((images * 127.5) + 127.5).astype("uint8")
    # images = np.repeat(images, 3, axis=-1)
    vis = build_montages(images, (gen_imgs.shape[1], gen_imgs.shape[2]), (row, col))[0]
    IMAGE_FILENAME = 'Results/test_{}_{}_output.png'.format(str(shape), str(count))
    cv2.imwrite(IMAGE_FILENAME, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
