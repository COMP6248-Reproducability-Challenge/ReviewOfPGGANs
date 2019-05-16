'''
File    : GAN.py
Author  : Jay Santokhi, Percy Jacks and Adrian Wilczynski (jks1g15, pj2g15, aw11g15)@soton.ac.uk
Brief   : Defines GAN class to build the TF graphs
Note    : Inspiration from:
          'GANs in action, Chapter 6: PGGAN
          https://github.com/GANs-in-Action/gans-in-action/tree/master/chapter-6
          and
          Mylittlerapture -- https://github.com/Mylittlerapture/GANLib
'''
import tensorflow as tf
import numpy as np
import logging
import time


class GAN(object):
    def __init__(self, sess, input_shape, latent_dim=100, optimizer=None, n_critic=1):
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        self.best_model = None
        self.best_metric = np.inf

        self.history = None

        self.epoch = tf.Variable(0)
        self.epochs = tf.Variable(0)

        self.optimizer = optimizer

        self.set_models_params()

        self.metric_func = self.magic_distance

        self.n_critic = n_critic
        self.sess = sess

    # Distance defined as (1 - average of probabilities data points from one set
    # appears in other set). The higher p and amount of data points, the better
    # the estimate
    def magic_distance(set_real, set_pred, p = 1000):

        set_pred_ = np.expand_dims(set_pred, axis=-1)
        set_real_ = np.expand_dims(set_real, axis=-1)
        dists = np.linalg.norm(set_pred_ - set_real_, axis = -1) ** (1/p)
        dists = dists.reshape((dists.shape[0], -1))

        result = (np.mean(dists, axis = -1) / np.amax(dists, axis = -1)) ** p
        return result

    def metric_test(self, set, pred_num = 32):
        '''
        Using magic distance as metric
        '''
        met_arr = np.zeros(pred_num)

        n_indx = np.random.choice(set.shape[0],pred_num)
        org_set = set[n_indx]

        noise = np.random.uniform(-1, 1, (pred_num, self.latent_dim))
        gen_set = self.predict(noise)
        met_arr = self.metric_func(org_set, gen_set)
        return met_arr

    def set_models_params(self):
        if self.optimizer is None: self.optimizer = tf.train.AdamOptimizer(0.001, 0.5, epsilon = 1e-07)
        self.models = ['generator', 'discriminator']

    def build_graph(self):
        '''
        Setting up losses and optimisers
        '''
        self.genr_input = tf.placeholder(tf.float32, shape=(None, self.latent_dim))
        self.disc_input = tf.placeholder(tf.float32, shape=(None,) + self.input_shape)

        # Generator loss variable set up
        def G(x):
            with tf.variable_scope('G', reuse=tf.AUTO_REUSE) as scope:
                return self.generator(x)

        self.genr = G(self.genr_input)

        # Discriminator loss variable set up
        def D(x):
            with tf.variable_scope('D', reuse=tf.AUTO_REUSE) as scope:
                return self.discriminator(x)

        d_real = D(self.disc_input)
        d_fake = D(self.genr)

        # Mean squared error loss function
        d_loss_real = tf.reduce_mean(tf.losses.mean_squared_error(predictions=d_real, labels=tf.ones_like(d_real)))
        d_loss_fake = tf.reduce_mean(tf.losses.mean_squared_error(predictions=d_fake, labels=tf.zeros_like(d_fake)))

        self.disc_loss = tf.reduce_mean(0.5*(d_loss_real + d_loss_fake))
        self.genr_loss = tf.reduce_mean(tf.losses.mean_squared_error(predictions=d_fake, labels=tf.ones_like(d_fake)))

        # Get train sessions
        vars_g = tf.trainable_variables('G')
        vars_d = tf.trainable_variables('D')
        self.train_genr = self.optimizer.minimize(self.genr_loss, var_list=vars_g)
        self.train_disc = self.optimizer.minimize(self.disc_loss, var_list=vars_d)

    def prepare_data(self, data_set, validation_split, batch_size):
        if 0. < validation_split < 1.:
            split_at = int(data_set.shape[0] * (1. - validation_split))
            self.train_set = data_set[:split_at]
            self.valid_set = data_set[split_at:]
        else:
            self.train_set = data_set
            self.valid_set = None

    def predict(self, noise, moving_average = False):
        if moving_average:
            imgs = self.sess.run(self.smooth_genr, feed_dict = {self.genr_input: noise})
        else:
            imgs = self.sess.run(self.genr, feed_dict = {self.genr_input: noise})
        return imgs

    def train_on_batch(self, batch_size):
        '''
        Training using random batches of images
        '''
        for j in range(self.n_critic):
            # Select a random batch of images
            idx = np.random.randint(0, self.train_set.shape[0], batch_size)
            imgs = self.train_set[idx]

            # Sample noise as generator input
            noise = np.random.uniform(-1, 1, (batch_size, self.latent_dim))
            self.sess.run(self.train_disc, feed_dict={self.disc_input: imgs, self.genr_input: noise})

        noise = np.random.uniform(-1, 1, (batch_size, self.latent_dim))
        self.sess.run([self.train_genr], feed_dict={self.disc_input: imgs, self.genr_input: noise})

        d_loss, g_loss = self.sess.run([self.disc_loss, self.genr_loss], feed_dict={self.disc_input: imgs, self.genr_input: noise})
        return d_loss, g_loss

    def build_models(self, files = None, custom_objects = None):
        for model in self.models:
            if not hasattr(self, model): raise Exception("%s are not defined!"%(model))

        self.build_graph()

        #Smooth generator
        ema = tf.train.ExponentialMovingAverage(decay = 0.999)
        def ema_getter(getter, name, *args, **kwargs):
            var = getter(name, *args, **kwargs)
            ema_var = ema.average(var)
            return ema_var if ema_var else var

        def Smooth_G(x):
            with tf.variable_scope('G', reuse=tf.AUTO_REUSE, custom_getter = ema_getter):
                res = self.generator(x)
            return res

        self.smooth_genr = Smooth_G(self.genr_input)

        #Initialize new variables
        vars = tf.global_variables()
        unint_vars_names = self.sess.run(tf.report_uninitialized_variables(vars))
        unint_vars_names = [u.decode("utf-8") for u in unint_vars_names]
        unint_vars = [ v for v in tf.global_variables() if v.name.split(':')[0] in unint_vars_names]

        self.sess.run(tf.variables_initializer(unint_vars))

        with tf.control_dependencies([self.train_genr]):
            with tf.variable_scope('', reuse=tf.AUTO_REUSE):
                self.train_genr = ema.apply(tf.trainable_variables('G'))

        vars = tf.global_variables()
        unint_vars_names = self.sess.run(tf.report_uninitialized_variables(vars))
        unint_vars_names = [u.decode("utf-8") for u in unint_vars_names]
        unint_vars = [ v for v in tf.global_variables() if v.name.split(':')[0] in unint_vars_names]

        self.sess.run(tf.variables_initializer(unint_vars))

    def test_network(self, batch_size):
        metric = self.metric_test(self.train_set, batch_size)
        return {'metric': metric}

    def train(self, data_set, batch_size=32, epochs=1, verbose=True, checkpoint_range=100,
        checkpoint_callback=None, validation_split=0, save_best_model=False, collect_history=True):

        # mean min max
        max_hist_size = epochs//checkpoint_range + 1
        history = { 'best_metric':0,
                    'hist_size'  :0}

        self.epoch.load(0, self.sess)
        self.epochs.load(epochs, self.sess)

        # Build Network
        self.prepare_data(data_set, validation_split, batch_size)
        self.build_models()

        # log the training run
        LOG_FILENAME = '{}_{}.log'.format('pggan', 'celeba')
        logging.basicConfig(filename='log' + '/' + LOG_FILENAME, level=logging.DEBUG)

        t = time.time()
        # Train Network
        for epoch in range(epochs):
            self.epoch.load(epoch, self.sess)

            d_loss, g_loss = self.train_on_batch(batch_size)

            # Save history
            if epoch % checkpoint_range == 0:
                d_t = time.time() - t
                t = time.time()

                # log the epoch stats
                logging.debug('{},{},{:.6f},{:.6f},{:.3f}'.format(epoch+1, epochs, d_loss, g_loss, d_t))

                if not collect_history:
                    if verbose: print('%d [D loss: %f] [G loss: %f] time: %f' % (epoch, d_loss, g_loss, d_t))
                else:
                    dict_of_vals = self.test_network(128)
                    dict_of_vals['D loss'] = d_loss
                    dict_of_vals['G loss'] = g_loss

                    hist_size = history['hist_size'] = history['hist_size']+1
                    metric = np.mean(dict_of_vals['metric'])

                    for k, v in dict_of_vals.items():
                        if k not in history:
                            history[k] = np.zeros((max_hist_size,3))

                        history[k][hist_size-1] = np.mean(v),  np.min(v),  np.max(v)

                    if verbose: print ("%d [D loss: %f] [G loss: %f] [%s: %f] time: %f" % (epoch, d_loss, g_loss, 'metric', metric, d_t))


                    if metric < self.best_metric:
                        self.best_metric = metric
                        history['best_metric'] = self.best_metric

                    self.history = history

                if checkpoint_callback is not None:
                    checkpoint_callback(epoch, data_set.shape[1])

        if save_best_model:
            self.generator.set_weights(self.best_model)

        self.epoch.load(epochs, self.sess)
        checkpoint_callback(epoch, data_set.shape[1])
        return self.history
