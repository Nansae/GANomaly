import os, time, math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from config import Config

class GANnomaly(object):
    def __init__(self, config):
        self.BATCH_SIZE = config.BATCH_SIZE
        self.latent_dim = config.LATENT_DIM
        self.width = config.IMG_SIZE
        self.height = config.IMG_SIZE
        self.channel = 1
        self.global_step = None
        if config.MODE == "train":
            self.global_step = tf.Variable(initial_value = 0, name='global_step', trainable=False, collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        self.image = tf.placeholder(tf.float32, [None, self.height, self.width, self.channel], name='real_images')
        #self.noise = tf.placeholder(tf.float32, [None, 1, 1, self.latent_dim], name='z')

        self.encode_data = self.encoder(self.image)
        self.gene = self.generator(self.encode_data)
        self.encode_gene = self.encoder(self.gene, reuse=True)
        self.feature_real, self.real = self.discriminator(self.image)
        self.feature_fake, self.fake = self.discriminator(self.gene, reuse=True)

        self.loss_G_context = tf.reduce_mean(tf.abs(self.gene - self.image))
        self.loss_G_encoder = tf.reduce_mean(tf.squared_difference(self.encode_gene, self.encode_data))
        self.loss_G_adv = tf.reduce_mean(tf.squared_difference(self.feature_fake, self.feature_real))

        #Generator Loss
        self.loss_G = 0.5 * self.loss_G_adv + 50 * self.loss_G_context + 1 * self.loss_G_encoder

        # use Sigmoid, Please not confused
        self.loss_D_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real, labels=tf.ones_like(tf.nn.sigmoid(self.real))))
        self.loss_D_gene = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake, labels=tf.zeros_like(tf.nn.sigmoid(self.fake))))
        
        #Discriminator Loss
        self.loss_D = self.loss_D_real + self.loss_D_gene

        self.vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        self.vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')

        self.encoded_input = self.encoder(self.image, is_training=False, reuse=True)
        self.sample = self.generator(self.encoded_input, is_training=False, reuse=True)
        self.encoded_sample = self.encoder(self.sample, is_training=False, reuse=True)

        print("Done building")

    def encoder(self, input, is_training=True, reuse=False):
        with tf.variable_scope('encoder', reuse=reuse) as scope:
            batch_norm_params = {'decay': 0.9,
                                 'epsilon': 0.001,
                                 'is_training': is_training,
                                 'scope': 'batch_norm'}
            with slim.arg_scope([slim.conv2d],
                                kernel_size = [4, 4],
                                stride = [2, 2],
                                activation_fn=tf.nn.leaky_relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params):

                net = slim.conv2d(input, 64, normalizer_fn=None, scope='layer1')
                net = slim.conv2d(net, 128, scope='layer2')
                net = slim.conv2d(net, 128, scope='layer3')
                net = slim.conv2d(net, 256, scope='layer4')
                net = slim.conv2d(net, 256, scope='layer5')
                net = slim.conv2d(net, 512, scope='layer6')
                net = slim.conv2d(net, 512, scope='layer7')
                net = slim.conv2d(net, self.latent_dim, kernel_size=[2, 2], stride=[1, 1], padding='VALID', normalizer_fn=None, activation_fn=None, scope='layer8')

                return net

    def generator(self, input, is_training=True, reuse=False):
        with tf.variable_scope('generator', reuse=reuse) as scope:
            batch_norm_params = {'decay': 0.9,
                                 'epsilon': 0.001,
                                 'is_training': is_training,
                                 'scope': 'batch_norm'}
            with slim.arg_scope([slim.conv2d_transpose],
                                kernel_size=[4, 4],
                                stride=[2, 2],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params):

                # inputs : 1 x 1x100
                net = slim.conv2d_transpose(input, 512, scope='layer1') # 2, 2, 256
                net = slim.conv2d_transpose(net, 512, scope='layer2') # 4, 4, 256
                net = slim.conv2d_transpose(net, 256, scope='layer3') # 8, 8, 256
                net = slim.conv2d_transpose(net, 256, scope='layer4') # 16, 16, 128
                net = slim.conv2d_transpose(net, 128, scope='layer5') # 32, 32, 64
                net = slim.conv2d_transpose(net, 128, scope='layer6') # 64, 64, 64
                net = slim.conv2d_transpose(net, 64, scope='layer7') # 128, 128, 32
                net = slim.conv2d_transpose(net, 1, normalizer_fn=None, activation_fn=tf.nn.tanh, scope='layer8') # 256, 256, 1

                return net

    def discriminator(self, input, is_training=True, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse) as scope:
            batch_norm_params = {'decay': 0.9,
                                 'epsilon': 0.001,
                                 'is_training': is_training,
                                 'scope': 'batch_norm'}
            with slim.arg_scope([slim.conv2d],
                                kernel_size = [4, 4],
                                stride = [2, 2],
                                activation_fn=tf.nn.leaky_relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params):

                net = slim.conv2d(input, 64, normalizer_fn=None, scope='layer1')
                net = slim.conv2d(net, 128, scope='layer2')
                net = slim.conv2d(net, 128, scope='layer3')
                net = slim.conv2d(net, 256, scope='layer4')
                net = slim.conv2d(net, 256, scope='layer5')
                net = slim.conv2d(net, 512, scope='layer6')
                net = slim.conv2d(net, 512, scope='layer7')
                feature = net
                net = slim.conv2d(net, 1, kernel_size=[2, 2], stride=[1, 1], padding='VALID', normalizer_fn=None, activation_fn=None, scope='layer8')
                logits = tf.squeeze(net, axis=[1, 2])

                return feature, logits

if __name__ == "__main__":
    model = GANnomaly(Config())