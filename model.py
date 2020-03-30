"""
Author: Kourosh T. Baghaei
A customized Implementation of U-Net for Image Segmentation Task
March 29, 2020

"""

import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Input,Activation
from tensorflow.python.keras.models import Model,Sequential
from tensorflow import keras

#
#  Network Structure:
#  Contractive Path: Layer 1 -> Layer 2 -> Layer 3
#  Bottleneck: Layer 4
#  Expanding Path: Layer 5 -> Layer 6 -> Layer 7
#


class MyUNet(tf.keras.Model):
    def __init__(self, input_shape=(128, 128, 3)):
        super(MyUNet, self).__init__()

        inputs = Input(shape=input_shape)

        # Layer 1
        self.layer1 = self.build_layer1(inputs)     #  layer1's output: (None, 170, 170, 64). First dimension, denoted
                                                    #   with 'None' represents the batch size.

        # Layer 2
        i2 = Input(self.layer1.output.shape[1:])
        self.layer2 = self.build_layer2(i2)         # layer2's output: (None, 83, 83, 128)

        # Layer 3
        i3 = Input(self.layer2.output.shape[1:])
        self.layer3 = self.build_layer3(i3)         # layer3's output: (None, 39, 39, 256)

        # Layer 4
        i4 = Input(self.layer3.output.shape[1:])
        self.layer4 = self.build_layer4(i4)         # layer4's output: (None, 34, 34, 256)

        # Layer 5
        s = self.layer4.output.shape
        self.crop_size = s[1:]      # output of layer3 should be cropped in heights and widths to match those of layer4.
        s = s[1:-1] + [s[-1] * 2]   # output of layer4 and layer3 are merged before being fed to layer5.
                                    # This is why the last dimension of the tensor is doubled.
                                    # layer5 input: (None, 34, 34, 512)
        self.layer5 = self.build_layer5(Input(s))   # layer5's output: (None, 64, 64, 128)

        # Layer 6
        s = self.layer5.output.shape
        s = s[1:-1] + [s[-1] * 2]   # output of layer 5 and layer 2 are merged together. So, the input of layer 6
                                    # looks like: (None, 64, 64, 128)
        self.layer6 = self.build_layer6(Input(s)) # layer6's output: (None, 124, 124, 64)

        # Layer 7
        s = self.layer6.output.shape
        s = s[1:-1] + [s[-1] * 2]   # output of layer 6 and layer 1 are merged together. So, the input of layer 7
                                    # looks like: (None, 124, 124, 128)
        self.layer7 = self.build_layer7(Input(s))   # layer7's output: (None, 122, 122, 2)

    def call(self, inputs):
        o1 = self.layer1(inputs)
        o2 = self.layer2(o1)
        o3 = self.layer3(o2)
        o4 = self.layer4(o3)

        # randomly crop o3 tensor in batches
        self.crop_size = o4.shape[1:]
        cropped = tf.map_fn(self.crop_rand, o3)
        # Skip Connection ( concat of layer3 and layer4 )
        i5 = layers.concatenate([cropped, o4])
        o5 = self.layer5(i5)

        # randomly crop o2 tensor in batches
        self.crop_size = o5.shape[1:]
        cropped = tf.map_fn(self.crop_rand, o2)
        # Skip Connection ( concat of layer2 and layer5 )
        i6 = layers.concatenate([cropped, o5])
        o6 = self.layer6(i6)

        # randomly crop o1 tensor in batches
        self.crop_size = o6.shape[1:]
        cropped = tf.map_fn(self.crop_rand, o1)
        # Skip Connection ( concat of layer1 and layer6 )
        i7 = layers.concatenate([cropped, o6])
        o7 = self.layer7(i7)

        return o7

    # randomly crop an image
    def crop_rand(self, img):
        return tf.image.random_crop(img, self.crop_size)

    def build_layer1(self, inp):
        layer1 = Sequential([
            layers.Conv2D(64, 2),
            layers.LeakyReLU(),
            layers.BatchNormalization(),
            layers.Conv2D(64, 2),
            layers.LeakyReLU(),
        ])(inp)
        print('layer 1', layer1.shape)
        return keras.Model(inp, layer1)

    def build_layer2(self, inp):
        layer2 = Sequential([
            layers.MaxPooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Conv2D(128, 2),
            layers.LeakyReLU(),
            layers.BatchNormalization(),
            layers.Conv2D(128, 2),
            layers.LeakyReLU(),
        ])(inp)
        print('layer 2', layer2.shape)
        return keras.Model(inp, layer2)

    def build_layer3(self, inp):
        layer3 = Sequential([
            layers.MaxPooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Conv2D(256, 2),
            layers.LeakyReLU(),
            layers.BatchNormalization(),
            layers.Conv2D(256, 2),
            layers.LeakyReLU(),
        ])(inp)
        print('layer 3', layer3.shape)
        return keras.Model(inp, layer3)

    def build_layer4(self, inp):
        layer4 = Sequential([
            layers.MaxPooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Conv2D(512, 2),
            layers.LeakyReLU(),
            layers.BatchNormalization(),
            layers.Conv2D(512, 2),
            layers.LeakyReLU(),
            layers.Dropout(0.3),
            layers.Conv2DTranspose(256, 2, 2),
        ])(inp)
        print('layer 4', layer4.shape)
        return keras.Model(inp, layer4)

    def build_layer5(self, inp):
        layer5 = Sequential([
            layers.Conv2D(512, 2),
            Activation('relu'),
            layers.Conv2D(256, 2),
            Activation('relu'),
            layers.Conv2DTranspose(128, 2, 2)
        ])(inp)
        print('layer 5 ', layer5.shape)
        return keras.Model(inp, layer5)

    def build_layer6(self, inp):
        layer6 = Sequential([
            layers.Conv2D(256, 2),
            Activation('relu'),
            layers.Conv2D(128, 2),
            Activation('relu'),
            layers.Conv2DTranspose(64, 2, 2)
        ])(inp)
        print('layer 6 ', layer6.shape)
        return keras.Model(inp, layer6)

    def build_layer7(self, inp):
        layer7 = Sequential([
            layers.Conv2D(256, 2),
            Activation('relu'),
            layers.Conv2D(128, 2),
            Activation('relu'),
            layers.Conv2D(2, 1),
            layers.Softmax()
        ])(inp)
        print('layer 7 ', layer7.shape)
        return keras.Model(inp, layer7)

