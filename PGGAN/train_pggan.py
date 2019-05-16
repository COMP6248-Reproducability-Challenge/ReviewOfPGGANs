'''
File    : train_pggan.py
Authors : Jay Santokhi, Percy Jacks and Adrian Wilczynski (jks1g15, pj2g15, aw11g15)@soton.ac.uk
Brief   : Trains a PGGAN
Note    : Inspiration from:
          'GANs in action, Chapter 6: PGGAN
          https://github.com/GANs-in-Action/gans-in-action/tree/master/chapter-6
          and
          Mylittlerapture -- https://github.com/Mylittlerapture/GANLib
'''
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pggan
import time

from scipy.misc import imresize
from utilities import sample_images
from utilities import augment
from glob import glob
from gan import GAN

row, col = 10, 10
noise_dim = 64

sheets = 0

# Load the dataset
# (dataseta, labelsa), (datasetb, labelsb) = tf.keras.datasets.cifar10.load_data()
# dataset = np.concatenate((dataseta, datasetb), axis = 0)
# labels = np.concatenate((labelsa, labelsb), axis = 0)

# Selecting the class in the dataset 0: plane, 1: cars, 2: bird, 3:cat, 4:deer
# 5: dog, 6: frog, 7: horse, 8: ship, 9: truck
# indx = np.where(labels == 7)[0]
# dataset = dataset[indx]

#-------------------------------------------------------------------------------
no_train = 10000

filenames = np.array(glob("img_align_celeba/*.jpg"))

print(filenames.shape)

X_train = []

size = (32, 32)

for i in range(no_train):
    img = plt.imread(filenames[i])
    # crop
    rows, cols = img.shape[:2]
    crop_r, crop_c = 150, 150
    start_row, start_col = (rows - crop_r) // 2, (cols - crop_c) // 2
    end_row, end_col = rows - start_row, cols - start_row
    img = img[start_row:end_row, start_col:end_col, :]
    # resize
    img = imresize(img, size)
    X_train.append(img)

X_train = np.array(X_train)
print("X_train.shape = {}".format(X_train.shape))

dataset = X_train
#-------------------------------------------------------------------------------

# Configure input
dataset = (dataset.astype(np.float32) - 127.5) / 127.5
if len(dataset.shape)<4:
    dataset = np.expand_dims(dataset, axis=3)

# Augment dataset to increase samples
# dataset = augment(dataset)

epochs_list = [4000, 8000, 16000, 32000]
batch_size_list = [16, 16, 16, 16]
image_size_list = [4, 8, 16, 32]

# Hyperparameters from paper
optimizer = tf.train.AdamOptimizer(0.001, 0., 0.99, epsilon = 1e-08)

# Training the network
with tf.Session() as sess:
    t = time.time()
    dataset_t = tf.Variable(np.zeros_like(dataset), dtype = tf.float32)
    for i in range(len(epochs_list)):
        epochs = epochs_list[i]
        batch_size = batch_size_list[i]

        data_set = sess.run(tf.image.resize_bilinear(dataset_t, (image_size_list[i],
                    image_size_list[i])), feed_dict = {dataset_t: dataset})
        print(data_set.shape)

        # Build and train GAN
        gan = GAN(sess, data_set.shape[1:], noise_dim, optimizer=optimizer)
        gan.generator = lambda x: pggan.generator(x, noise_dim, sheets, gan)
        gan.discriminator = lambda x: pggan.discriminator(x, sheets, gan)

        # Generated images defined as a callback
        def callback(int, shape):
            sample_images(gan, row, col, noise_dim, 'pg_gan.png', int, shape)
        gan.train(data_set, epochs=epochs, batch_size=batch_size, checkpoint_callback=callback,
                    collect_history=False)
        sheets += 1

    print('Training complete! Total training time: %f s'%(time.time() - t))
