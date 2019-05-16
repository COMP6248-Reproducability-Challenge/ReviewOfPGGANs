'''
File    : pggan.py
Author  : Jay Santokhi, Percy Jacks and Adrian Wilczynski (jks1g15, pj2g15, aw11g15)@soton.ac.uk
Brief   : Defines the Generator and Discriminator for PGGAN
Note    : Inspiration from:
          'GANs in action, Chapter 6: PGGAN
          https://github.com/GANs-in-Action/gans-in-action/tree/master/chapter-6
          and
          Mylittlerapture -- https://github.com/Mylittlerapture/GANLib
'''
import tensorflow as tf
import numpy as np
import keras as K

from utilities import equalise_learning_rate
from utilities import minibatch_std_layer
from utilities import transition_alpha
from utilities import upscale_layer
from utilities import new_sheet

# Parameters for training
initialization = equalise_learning_rate
channels = 3
filters_list = [48, 32, 24, 16]
image_size_list = [4, 8, 16, 32]

def generator(input, noise_dim, sheets, gan):
    '''
    Function that builds the Generator part of the GAN architecture

    Arguments:
        input - dimension of the randomly generated input vector
        noise_dim - dimension of the noise
        sheets - variable used in growing layers
        gan - used for the threshold alpha (for smoothly merging last layer)

    Returns:
        layer - generator layer
    '''
    previous_step = None
    next_step = None

    layer = input

    layer = tf.keras.layers.RepeatVector(16)(layer)
    layer = tf.keras.layers.Reshape((4, 4, noise_dim))(layer)

    layer = new_sheet(layer, filters_list[0], (4,4), 'same', 'genr_head_0')
    layer = new_sheet(layer, filters_list[0], (3,3), 'same', 'genr_head_1')

    # Growing layers
    for i in range(sheets):
        s = image_size_list[i + 1]
        layer = upscale_layer(layer)
        if i == sheets-1: previous_step = layer

        layer = new_sheet(layer, filters_list[i+1], (3,3), 'same', 'genr_layer_a'+str(i))
        layer = new_sheet(layer, filters_list[i+1], (3,3), 'same', 'genr_layer_b'+str(i))

    # to RGB
    next_step = tf.layers.conv2d(layer, channels, (1,1), name='to_rgb_'+str(sheets),
                kernel_initializer = initialization)

    # smooth fading
    if previous_step is not None:
        previous_step = tf.layers.conv2d(previous_step, channels, (1,1), name='to_rgb_'+str(sheets - 1))
        layer = previous_step + (next_step - previous_step) * transition_alpha(gan)
    else:
        layer = next_step

    return layer


def discriminator(input, sheets, gan):
    '''
    Function that builds the Discriminator part of the GAN architecture

    Arguments:
        input - dimension of the randomly generated input vector
        sheets - variable used in growing layers
        gan - used for the threshold alpha (for smoothly merging last layer)

    Returns:
        layer - discriminator layer
    '''
    previous_step = None
    next_step = None

    input_layer = input

    # from RGB
    layer = tf.layers.conv2d(input, filters_list[sheets], (1,1), name='from_rgb_'+str(sheets),
                            kernel_initializer=initialization)
    layer = tf.nn.leaky_relu(layer, alpha=0.2)

    # Growing layers
    for i in range(sheets, 0, -1):
        layer = new_sheet(layer, filters_list[i], (3,3), 'same', 'disc_layer_b'+str(i), pix_norm=False)
        layer = new_sheet(layer, filters_list[i - 1], (3,3), 'same', 'disc_layer_a'+str(i), pix_norm=False)
        layer = tf.layers.average_pooling2d(layer, 2, 2)

        # smooth fading
        if i == sheets:
            next_step = layer

            previous_step = tf.layers.average_pooling2d(input_layer, 2, 2)
            # from RGB
            previous_step = tf.layers.conv2d(previous_step, filters_list[i-1], (1,1),
                            name='from_rgb_'+str(sheets-1), kernel_initializer=initialization)
            previous_step = tf.nn.leaky_relu(previous_step, alpha=0.2)

            layer=previous_step + (next_step - previous_step) * transition_alpha(gan)

    layer = minibatch_std_layer(layer)
    layer = new_sheet(layer, filters_list[0], (3,3), 'same', 'disc_head_0', pix_norm=False)
    layer = new_sheet(layer, filters_list[0], (4,4), 'valid', 'disc_head_1', pix_norm=False)

    layer = tf.keras.layers.Flatten()(layer)
    layer = tf.layers.dense(layer, 1, kernel_initializer=initialization)

    return layer
