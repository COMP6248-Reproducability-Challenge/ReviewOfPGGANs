'''
File    : gan.py
Authors : Jay Santokhi, Adrian Wilczynski, Percy Jacks (jks1g15, aw11g15, pj2g15)@soton.ac.uk
Brief   : Creates a class for a generic GAN using methods by Radford et al
'''
import os
# run on plaidml if installed, else use TF. (plaidml does not work on all datasets and all classes as is still in development)
# try:
#     import plaidml
#     os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
# except ImportError:
#     pass

import keras
from keras import backend as K
from keras.layers import BatchNormalization
from keras.layers import Conv2DTranspose
from keras.layers import Conv2D
from keras.layers import LeakyReLU
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Dropout
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from scipy.misc import imresize

from sklearn.utils import shuffle
from imutils import build_montages

from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.datasets import cifar10

import re
import numpy as np
import cv2
import time
import logging
import datetime
import matplotlib.pyplot as plt
from glob import glob

# print('tensorflow version = ', tf.VERSION)
# print('keras version = ', tf.keras.__version__)


class GAN():
    '''
    Instanciate a GAN with the sepecified configuration. Default is a DCGAN using the MNIST dataset.

    Arguments:
    gantype - Select the type of gan to use. Can be 'dcgan' or 'lsgan'
    '''

    def __init__(self, gantype='dcgan', dataset='mnist'):
        self.gantype = 0
        self.dataset = 'NOT_ASSIGNED'
        self.trainX = 0
        self.trainY = 0
        self.testX = 0
        self.testY = 0
        self.trainset = 0
        self.compiled_gan = 0
        self.chosen_class = ''
        if(re.search(gantype, "dcgan", flags=re.IGNORECASE)):
            self.gantype = 'dcgan'
            self.disc_loss = 'binary_crossentropy'
        elif(re.search(gantype, "wgan", flags=re.IGNORECASE)):
            self.gantype = 'wgan'
            self.disc_loss = self.wasserstein_loss
        elif(re.search(gantype, "lsgan", flags=re.IGNORECASE)):
            self.gantype = 'lsgan'
            self.disc_loss = 'mse'

    def wasserstein_loss(self, y_label, y_pred):
        "Wasserstein loss function. Not able to make work, only produced noise."
        return -K.mean(y_label * y_pred)

    def load_dataset(self, dataset='cifar10', set_class='horse', augmentation='True'):
        '''
        Function that loads a dataset. If cifar10 is chosen, a class must also be
        chosen on which to train, for example 'horse' or 'car'. Also allows the
        augmentation of cifar10 dataset with the augmentation argument set to 'True'

        Arguments:
            dataset - the chosen dataset to be used for training.
            set_class - a class in the dataset (only used for cifar10).
            augmentation - allows for augmenting the class data.

        Returns:
            trainset - the portion of the dataset used for training in the format:
                       ((trainX, trainY),(testX, testY))

        Notes:
            cifar10 classes to choose from:
            plane, car, bird, cat, deer, dog, frog, horse, ship, truck

        '''

        # loads the cifar10 dataset if it is chosen
        if(re.search(dataset, "cifar10", flags=re.IGNORECASE)):
            self.dataset = 'cifar10'

            # the below returns in the format: ((trainX, trainY), (testX, testY))
            ((self.trainX, self.trainY), (self.testX, self.testY)) = cifar10.load_data()  # 32x32x3
            self.chosen_class = set_class
            if(set_class == 'plane'):
                class_index = 0
            elif(set_class == 'car'):
                class_index = 1
            elif(set_class == 'bird'):
                class_index = 2
            elif(set_class == 'cat'):
                class_index = 3
            elif(set_class == 'deer'):
                class_index = 4
            elif(set_class == 'dog'):
                class_index = 5
            elif(set_class == 'frog'):
                class_index = 6
            elif(set_class == 'horse'):
                class_index = 7
            elif(set_class == 'ship'):
                class_index = 8
            elif(set_class == 'truck'):
                class_index = 9
            else:
                print('\nERROR: Class not found in dataset\n')
                exit()

            index = (self.trainY == class_index)
            index2 = (self.testY == class_index)

            trainX = self.trainX[index[:, 0]]
            testX = self.testX[index2[:, 0]]

            # concatenates both the train and test sections of the dataset to use the entire dataset since testing is not required.
            trainset = np.concatenate([trainX, testX])
            print(trainset.shape)

            # normalising the trainset from a range of 0 to 255 to -1 to 1 for the tanh activation function in the generator.
            trainset = (trainset.astype("float") - 127.5) / 127.5
            print('dataset = {}'.format(self.dataset))
            print('class = {}'.format(self.chosen_class))

            # checks whether augmentation has been enabled, and augments the dataset to increase sample number
            if(augmentation == 'True'):
                print('cifar10 shape before augmentation:', trainset.shape)

                off_x_p = trainset.copy()
                off_x_p[:, 1:, :, :] = off_x_p[:, :-1, :, :]
                off_x_m = trainset.copy()
                off_x_p[:, :-1, :, :] = off_x_p[:, 1:, :, :]

                off_y_p = trainset.copy()
                off_y_p[:, :, 1:, :] = off_y_p[:, :, :-1, :]
                off_y_m = trainset.copy()
                off_y_m[:, :, :-1, :] = off_y_m[:, :, 1:, :]
                trainset = np.concatenate((trainset, off_y_p, off_x_p, off_y_m, off_x_m), axis=0)

                print('cifar10 shape after augmentation:', trainset.shape)

            else:
                print('cifar10 shape:', trainset.shape)

            return trainset

        # checks if mnist or fashion_mnist is being used
        elif(re.search(dataset, "mnist", flags=re.IGNORECASE) or re.search(dataset, "fashion_mnist", flags=re.IGNORECASE)):
            if(re.search(dataset, "mnist", flags=re.IGNORECASE)):
                self.dataset = 'mnist'
                # the below returns in the format: ((trainX, _), (testX, _))
                ((self.trainX, self.trainY), (self.testX, self.testY)) = mnist.load_data()  # 28x28x1

            elif(re.search(dataset, "fashion_mnist", flags=re.IGNORECASE)):
                self.dataset = 'fashion_mnist'
                # the below returns in the format: ((trainX, _), (testX, _))
                ((self.trainX, self.trainY), (self.testX, self.testY)) = fashion_mnist.load_data()  # 28x28x1

            trainset = np.concatenate([self.trainX, self.testX])
            print(trainset.shape)

            trainset = np.expand_dims(trainset, axis=-1)

            trainset = (trainset.astype("float") - 127.5) / 127.5
            print('Dataset \'{}\' loaded'.format(self.dataset))

            return trainset

        # checks if celeba is being used
        elif(re.search("^celeba", dataset, flags=re.IGNORECASE)):
            no_train = 120000

            filenames = np.array(glob('img_align_celeba/*.jpg'))

            plt.figure(figsize=(5, 4))

            X_train = []

            if(re.search("celeba128", dataset, flags=re.IGNORECASE)):
                self.dataset = 'celeba128'
                size = (128, 128)
            elif(re.search("celeba64", dataset, flags=re.IGNORECASE)):
                self.dataset = 'celeba64'
                size = (64, 64)
            else:
                self.dataset = 'celeba32'
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
            X_train = (X_train.astype("float") - 127.5) / 127.5
            print("X_train.shape = {}".format(X_train.shape))
            
            return X_train

        elif(re.search("pokemon", dataset, flags=re.IGNORECASE)):
            no_train = 1500

            filenames = np.array(glob('pokemon_sprites/*.png'))

            plt.figure(figsize=(5, 4))

            X_train = []
            
            # using the same network structure as the appropriately sized celeba network
            self.dataset = 'celeba64'
            size = (64, 64)

            for i in range(no_train):
                img = plt.imread(filenames[i])
                # crop
                rows, cols = img.shape[:2]
                crop_r, crop_c = 64, 64
                start_row, start_col = (rows - crop_r) // 2, (cols - crop_c) // 2
                end_row, end_col = rows - start_row, cols - start_row
                img = img[start_row:end_row, start_col:end_col, :]
                # resize
                img = imresize(img, size)
                img = img[:, :, :3]
                X_train.append(img)

            X_train = np.array(X_train)
            X_train = (X_train.astype("float") - 127.5) / 127.5

            # augmenting once
            off_x_p = X_train.copy()
            off_x_p[:, 1:, :, :] = off_x_p[:, :-1, :, :]
            off_x_m = X_train.copy()
            off_x_p[:, :-1, :, :] = off_x_p[:, 1:, :, :]

            off_y_p = X_train.copy()
            off_y_p[:, :, 1:, :] = off_y_p[:, :, :-1, :]
            off_y_m = X_train.copy()
            off_y_m[:, :, :-1, :] = off_y_m[:, :, 1:, :]
            X_train = np.concatenate((X_train, off_y_p, off_x_p, off_y_m, off_x_m), axis=0)

            # augmenting twice (if desired for a larger dataset)
            # off_x_p = X_train.copy()
            # off_x_p[:, 1:, :, :] = off_x_p[:, :-1, :, :]
            # off_x_m = X_train.copy()
            # off_x_p[:, :-1, :, :] = off_x_p[:, 1:, :, :]

            # off_y_p = X_train.copy()
            # off_y_p[:, :, 1:, :] = off_y_p[:, :, :-1, :]
            # off_y_m = X_train.copy()
            # off_y_m[:, :, :-1, :] = off_y_m[:, :, 1:, :]
            # X_train = np.concatenate((X_train, off_y_p, off_x_p, off_y_m, off_x_m), axis=0)
            print("pokemon X_train.shape = {}".format(X_train.shape))

            return X_train

    def generator(self, depth, inputDim, outputDim):
        '''
        Function that builds the Generator part of the GAN architecture

        Arguments:
            depth - Depth of the volume after reshaping
            inputDim - Dimension of the randomly generated input vector
            outputDim - Dimension of the output FC layer from generated input vector.

        Returns:
            model - The Generator model

        Notes:
            dim and channels are auto-assigned based on the dataset being used

            dim - Dimensions (width=dim, height=dim) of the generator after reshaping
            channels - 1 for greyscale, 3 for RGB
        '''
        if((self.dataset == 'mnist') or (self.dataset == 'fashion_mnist')):
            dim, channels = 7, 1
            model = keras.Sequential()

            inputShape = (dim, dim, depth)

            model.add(Dense(input_dim=inputDim, units=outputDim, activation='relu'))
            model.add(BatchNormalization())

            model.add(Dense(units=dim*dim*depth, activation='relu'))
            model.add(BatchNormalization())

            model.add(Reshape(inputShape))
            model.add(Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', activation='relu'))
            model.add(BatchNormalization())

            model.add(Conv2DTranspose(channels, (5, 5), strides=(2, 2), padding='same', activation='tanh'))

            return model

        elif((self.dataset == 'cifar10') or (self.dataset == 'celeba32')):
            dim, channels = 8, 3
            model = keras.Sequential()

            inputShape = (dim, dim, depth)

            model.add(Dense(input_dim=inputDim, units=outputDim, activation='relu'))
            model.add(BatchNormalization())

            model.add(Dense(units=dim*dim*depth, activation='relu'))
            model.add(BatchNormalization())

            model.add(Reshape(inputShape))
            model.add(Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',
                                      activation='relu'))
            model.add(BatchNormalization())

            model.add(Conv2D(32, (4, 4), strides=1, padding='same', activation='relu'))
            model.add(BatchNormalization())

            model.add(Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same', activation='relu'))
            model.add(BatchNormalization())

            model.add(Conv2D(channels, (4, 4), strides=1, padding='same', activation='tanh'))

            return model

        elif(self.dataset == 'celeba64'):
            # GENERATOR FOR CELEBA 64x64
            dim, channels = 8, 3
            model = keras.Sequential()

            inputShape = (dim, dim, depth)

            model.add(Dense(input_dim=inputDim, units=outputDim, activation='relu'))
            model.add(BatchNormalization())

            model.add(Dense(units=dim*dim*depth, activation='relu'))
            model.add(BatchNormalization())

            model.add(Reshape(inputShape))
            model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',
                                      activation='relu'))
            model.add(BatchNormalization())

            model.add(Conv2D(32, (4, 4), strides=1, padding='same', activation='relu'))
            model.add(BatchNormalization())

            model.add(Conv2DTranspose(32, (3, 3), strides=(4, 4), padding='same', activation='relu'))
            model.add(BatchNormalization())

            # model.add(Conv2D(16, (4, 4), strides=1, padding='same', activation='relu'))

            # model.add(Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same', activation='relu'))
            # model.add(BatchNormalization())

            model.add(Conv2D(channels, (4, 4), strides=1, padding='same', activation='tanh'))

            return model

        elif(self.dataset == 'celeba128'):
            pass

        else:
            print('\nERROR: Unknown dataset)')

    def discriminator(self, alpha):
        '''
        Function that builds the Discriminator part of the GAN architecture

        Arguments:
            alpha - Coefficient of the negative slope of Leaky-ReLU

        Returns:
            model - The Discriminator model

        Notes:
            width, height and depth are auto-assigned based on the dataset being used

            width - Width of the input to the Discriminator
            height - Height of the input to the Discriminator
            depth - Channels of the input to the Discriminator
        '''
        if((self.dataset == 'mnist') or (self.dataset == 'fashion_mnist')):
            width, height, channels = 28, 28, 1
            model = keras.Sequential()
            inputShape = (height, width, channels)

            model.add(Conv2D(32, (5, 5), padding='same', strides=(2, 2), input_shape=inputShape))
            model.add(LeakyReLU(alpha=alpha))

            model.add(Conv2D(64, (5, 5), padding='same', strides=(2, 2)))
            model.add(LeakyReLU(alpha=alpha))

            model.add(Flatten())
            model.add(Dense(512))
            model.add(LeakyReLU(alpha=alpha))

            model.add(Dense(1, activation='sigmoid'))

            return model

        elif((self.dataset == 'cifar10') or (self.dataset == 'celeba32')):
            width, height, channels = 32, 32, 3
            model = keras.Sequential()
            inputShape = (height, width, channels)

            model.add(Conv2D(32, (3, 3), padding='same', strides=(2, 2), input_shape=inputShape))
            model.add(LeakyReLU(alpha=alpha))
            model.add(Dropout(0.2))

            model.add(Conv2D(64, (4, 4), padding='same', strides=(2, 2)))
            model.add(LeakyReLU(alpha=alpha))
            model.add(Dropout(0.2))

            model.add(Conv2D(64, (4, 4), padding='same', strides=(2, 2)))
            model.add(LeakyReLU(alpha=alpha))
            model.add(Dropout(0.2))

            # model.add(Conv2D(32, (4, 4), padding='same', strides=(2, 2)))
            # model.add(LeakyReLU(alpha=alpha))
            # model.add(Dropout(0.2))

            model.add(Flatten())
            model.add(Dense(512))
            model.add(LeakyReLU(alpha=alpha))

            model.add(Dense(1, activation='sigmoid'))

            return model

        elif(self.dataset == 'celeba64'):
            # DISCRIMINATOR FOR CELEBA 64x64
            width, height, channels = 64, 64, 3
            model = keras.Sequential()
            inputShape = (height, width, channels)

            model.add(Conv2D(32, (3, 3), padding='same', strides=(2, 2), input_shape=inputShape))
            model.add(LeakyReLU(alpha=alpha))
            model.add(Dropout(0.2))

            model.add(Conv2D(32, (4, 4), padding='same', strides=(2, 2)))
            model.add(LeakyReLU(alpha=alpha))
            model.add(Dropout(0.2))

            model.add(Conv2D(64, (4, 4), padding='same', strides=(2, 2)))
            model.add(LeakyReLU(alpha=alpha))
            model.add(Dropout(0.2))

            model.add(Conv2D(64, (4, 4), padding='same', strides=(2, 2)))
            model.add(LeakyReLU(alpha=alpha))
            model.add(Dropout(0.2))

            model.add(Flatten())
            model.add(Dense(512))
            model.add(LeakyReLU(alpha=alpha))

            model.add(Dense(1, activation='sigmoid'))

            return model

        elif(self.dataset == 'celeba128'):
            pass

        else:
            print('\nERROR: Unknown dataset)')

    def build(self, Generator, Discriminator, epochs):
        print('Building GAN')
        GAN_input = Input(shape=(200,))
        GAN_output = Discriminator(Generator(GAN_input))
        GAN = Model(GAN_input, GAN_output)

        GAN_Optimiser = Adam(lr=0.0002, beta_1=0.5, decay=0.0002/epochs)
        GAN.compile(loss=self.disc_loss, optimizer=GAN_Optimiser)
        print(GAN.summary())

        self.compiled_gan = GAN

    def train(self, Generator, Discriminator, trainset, benchmarkNoise, epochs, batch_size):
        '''
        Function to train the gan object

        Arguments:
            GAN - a compiled gan object
        '''
        print('Training GAN...')

        average_epoch_secs = 0

        # create the folder directory to store the generated images
        now = datetime.datetime.now()
        now = (now.strftime("%d%m%Y_%H%M%S"))
        FOLDER = './{}_{}_{}_augmented_{}'.format(self.gantype, self.dataset, self.chosen_class, now)

        # create a directory
        try:
            os.mkdir(FOLDER)
            print('clean directory made')
        except OSError:
            print("creation of clean directory failed")

        # log the training run
        LOG_FILENAME = '{}_{}.log'.format(self.gantype, self.dataset)
        logging.basicConfig(filename=FOLDER + '/' + LOG_FILENAME, level=logging.DEBUG)

        time_start = time.time()
        for epoch in range(epochs):
            time_at_epoch_start = time.time()
            # print('Epoch {} of {}...'.format(epoch + 1, epochs))

            batchesPerEpoch = int(trainset.shape[0]/batch_size)

            for i in range(0, batchesPerEpoch):
                # Initialize an empty output path, p
                p = None

                # Select the next batch of images
                imageBatch = trainset[i * batch_size:(i + 1) * batch_size]

                # Randomly generate noise for the generator to predict on
                noise = np.random.normal(-1, 1, size=(batch_size, 200))

                # Generate images using the noise and generator model
                genImages = Generator.predict(noise, verbose=0)

                # Concatenate the real images and the fake images,
                # construct class labels for the discriminator, and shuffle
                # print(imageBatch.shape)
                # print(genImages.shape)
                X = np.concatenate((imageBatch, genImages))
                y = ([1] * batch_size) + ([0] * batch_size)
                (X, y) = shuffle(X, y)

                # Train the discriminator on the data
                discLoss = Discriminator.train_on_batch(X, y)

                # Train generator via the adversarial model by
                # generating random noise and training the generator
                # with the discriminator weights frozen
                noise = np.random.normal(-1, 1, (batch_size, 200))

                ganLoss = self.compiled_gan.train_on_batch(noise, [1] * batch_size)

                # Check to see if this is the end of an epoch, and if so,
                # initialize the output path
                if i == batchesPerEpoch - 1:
                    IMAGE_FILENAME = '{}_epoch_{}_output.png'.format(self.gantype, str(epoch + 1).zfill(4))
                    p = ['./' + FOLDER + '/' + IMAGE_FILENAME]

                if p is not None:
                    # Show loss information
                    print("\nEpoch {}/{}: discriminator_loss={:.6f}, adversarial_loss={:.6f}".format(epoch+1, epochs, discLoss, ganLoss), end='')

                    # Make predictions on the benchmark noise, scale it back
                    # to the range [0, 255], and generate the montage
                    images = Generator.predict(benchmarkNoise)
                    images = ((images * 127.5) + 127.5).astype("uint8")

                    if((self.dataset == 'mnist') or (self.dataset == 'fashion_mnist')):
                        images = np.repeat(images, 3, axis=-1)
                        vis = build_montages(images, (28, 28), (10, 10))[0]

                    elif((self.dataset == 'cifar10') or (self.dataset == 'celeba32')):
                        vis = build_montages(images, (32, 32), (10, 10))[0]

                    elif(self.dataset == 'celeba64'):
                        vis = build_montages(images, (64, 64), (10, 10))[0]

                    elif(self.dataset == 'celeba128'):
                        vis = build_montages(images, (128, 128), (10, 10))[0]

                    # write the visualization to disk
                    p = os.path.sep.join(p)
                    cv2.imwrite(p, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

            # log the epoch stats
            epoch_secs = (time.time()-time_at_epoch_start)
            logging.debug('{},{},{:.6f},{:.6f},{:.3f}'.format(epoch+1, epochs, discLoss, ganLoss, epoch_secs))
            if(epoch <= 1):
                average_epoch_secs = epoch_secs
            else:
                average_epoch_secs = ((average_epoch_secs * (epoch)) + epoch_secs)/(epoch+1)

            elapsed_hours = int((time.time()-time_start) / 3600)
            elapsed_mins = int(int((time.time()-time_start) / 60) - elapsed_hours*60) % 60

            eta_hours = int((average_epoch_secs*(epochs-(epoch+1)))/3600)
            eta_mins = int((average_epoch_secs*(epochs-(epoch+1)))/60)
            print('\nThis epoch took {:.2f} seconds  |  Avg epoch time:  {:.2f} seconds |  Time since start: {} h {} m  |  ETA: {} h {} m\n'.format(epoch_secs, average_epoch_secs, elapsed_hours, elapsed_mins, eta_hours, eta_mins-(eta_hours*60)))
            # reset running average after first epoch as is longer than others

        hours = int((time.time()-time_start) / 3600)
        mins = int(int((time.time()-time_start) / 60) - hours*60) % 60
        print('\nFinished training of {} {} took {} {} and {} {}'.format(epochs, ('epochs' if epochs > 1 else 'epoch'), hours, ('hour' if hours == 1 else 'hours'), mins, ('minute' if mins == 1 else 'minutes')))


if __name__ == "__main__":
    my_gan = GAN()
    print('passed checks')
