'''
File    : train_gan.py
Authors : Jay Santokhi, Adrian Wilczynski, Percy Jacks (jks1g15, aw11g15, pj2g15)@soton.ac.uk
Brief   : Trains the gan from the gan.py file
'''

import sys
args = sys.argv[1:]

# check for any arguments passed in executing the script
try:
    args[0]
    GANTYPE = args[0]
except IndexError:
    GANTYPE = 'lsgan'
try:
    args[1]
    DATASET = args[1]
except IndexError:
    DATASET = 'cifar10'
try:
    args[2]
    SET_CLASS = args[2]
except IndexError:
    SET_CLASS = 'horse'
try:
    args[3]
    AUGMENTATION = args[3]
except IndexError:
    AUGMENTATION = 'True'


import os
# run on plaidml if installed, else use TF. (plaidml does not work on all datasets and all classes as is still in development)
# try:
#     import plaidml
#     os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
# except ImportError:
#     pass

from keras.optimizers import Adam
import numpy as np

from gan import GAN

# create an instance of the gan and choose what type it is
my_gan = GAN(gantype=GANTYPE)
print('gantype = ', my_gan.gantype)

trainset = my_gan.load_dataset(dataset=DATASET, set_class=SET_CLASS, augmentation=AUGMENTATION)

# set up the training parameters. ----------------------------------------------
epochs = 500
batch_size = 64

new_gan = GAN

print('\nBuilding Generator')
# Generator = my_gan.generator(64, 100, 512)
Generator = my_gan.generator(64, 200, 512)
print('Generator built')
print(Generator.summary())

print('\nBuilding Discriminator')
Discriminator = my_gan.discriminator(0.2)
DisOptimiser = Adam(lr=0.0002, beta_1=0.5, decay=0.0002/epochs)
print('self.discloss', my_gan.disc_loss)
Discriminator.compile(loss=my_gan.disc_loss, optimizer=DisOptimiser)
print('Discriminator built')
print(Discriminator.summary())

Discriminator.trainable = False

my_gan.build(Generator, Discriminator, epochs)

# different noise sizes can improve performance of certain classes within cifar10, or datasets.
# benchmarkNoise = np.random.normal(-1, 1, size=(196, 100))
benchmarkNoise = np.random.normal(-1, 1, size=(100, 200))

my_gan.train(Generator, Discriminator, trainset, benchmarkNoise, epochs, batch_size)
