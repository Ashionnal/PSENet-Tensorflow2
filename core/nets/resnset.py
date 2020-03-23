import tensorflow as tf
from tensorflow import keras

class Block(keras.layers.Layer):
    def __init__(self,  filters_a, filters_b, kernel_size=(3, 3), padding='same', pool_size=(2, 2), is_maxpooling=False, filters=None):
        super(Block, self).__init__()
        self.is_maxpooling = is_maxpooling

        self.conv1 = keras.layers.Conv2D(filters=filters_a, kernel_size=kernel_size, strides=2, padding=padding)
        self.bn_conv1 = keras.layers.BatchNormalization()
        self.activation_1 = keras.layers.ReLU()
        self.max_pooling2d_1 = keras.layers.MaxPool2D(pool_size=pool_size)

        self.res2a_branch2a = keras.layers.Conv2D(filters=filters_a, kernel_size=kernel_size, strides=(1 if is_maxpooling else 2), padding=padding)
        self.bn2a_branch2a = keras.layers.BatchNormalization()
        self.activation_2 = keras.layers.ReLU()
        self.res2a_branch2b = keras.layers.Conv2D(filters=filters_a, kernel_size=kernel_size, strides=1, padding=padding)
        self.bn2a_branch2b = keras.layers.BatchNormalization()
        self.activation_3 = keras.layers.ReLU()
        self.res2a_branch2c = keras.layers.Conv2D(filters=filters_b, kernel_size=kernel_size, strides=1, padding=padding)
        self.bn2a_branch2c = keras.layers.BatchNormalization()

        self.res2a_branch1 = keras.layers.Conv2D(filters=filters_b, kernel_size=kernel_size, strides=(1 if is_maxpooling else 2), padding=padding)
        self.bn2a_branch1 = keras.layers.BatchNormalization()

        # add

        self.activation_4 = keras.layers.ReLU()

        self.res2b_branch2a = keras.layers.Conv2D(filters=filters_a, kernel_size=kernel_size, strides=1, padding=padding)
        self.bn2b_branch2a = keras.layers.BatchNormalization()
        self.activation_5 = keras.layers.ReLU()
        self.res2b_branch2b = keras.layers.Conv2D(filters=filters_a, kernel_size=kernel_size, strides=1, padding=padding)
        self.bn2b_branch2b = keras.layers.BatchNormalization()
        self.activation_6 = keras.layers.ReLU()
        self.res2b_branch2c = keras.layers.Conv2D(filters=filters_b, kernel_size=kernel_size, strides=1, padding=padding)
        self.bn2b_branch2c =keras.layers.BatchNormalization()
        
        # add

        self.activation_7 = keras.layers.ReLU()

        self.res2c_branch2a = keras.layers.Conv2D(filters=filters_a, kernel_size=kernel_size, strides=1, padding=padding)
        self.bn2c_branch2a = keras.layers.BatchNormalization()
        self.activation_8 = keras.layers.ReLU()
        self.res2c_branch2b = keras.layers.Conv2D(filters=filters_a, kernel_size=kernel_size, strides=1, padding=padding)
        self.bn2c_branch2b = keras.layers.BatchNormalization()
        self.activation_9 = keras.layers.ReLU()
        self.res2c_branch2c = keras.layers.Conv2D(filters=filters_b, kernel_size=kernel_size, strides=1, padding=padding)
        self.bn2c_branch2c = keras.layers.BatchNormalization()
        self.activation_10 = keras.layers.ReLU()

    def call(self, inputs, training=True):
        if self.is_maxpooling:
            outs = self.conv1(inputs)
            outs = self.bn_conv1(outs)
            outs = self.activation_1(outs)
            outs = self.max_pooling2d_1(outs)
        else:
            outs = inputs

        outs_branch1 = self.res2a_branch1(outs)
        outs_branch1 = self.bn2a_branch1(outs_branch1)

        outs_branch2 = self.res2a_branch2a(outs)
        outs_branch2 = self.bn2a_branch2a(outs_branch2)
        outs_branch2 = self.activation_2(outs_branch2)
        outs_branch2 = self.res2a_branch2b(outs_branch2)
        outs_branch2 = self.bn2a_branch2b(outs_branch2)
        outs_branch2 = self.activation_3(outs_branch2)
        outs_branch2 = self.res2a_branch2c(outs_branch2)
        outs_branch2 = self.bn2a_branch2c(outs_branch2)

        outs = tf.math.add_n((outs_branch2, outs_branch1))

        outs = self.activation_4(outs)

        outs_branch1 = outs
        outs_branch2 = self.res2b_branch2a(outs)
        outs_branch2 = self.bn2b_branch2a(outs_branch2)
        outs_branch2 = self.activation_5(outs_branch2)
        outs_branch2 = self.res2b_branch2b(outs_branch2)
        outs_branch2 = self.bn2b_branch2b(outs_branch2)
        outs_branch2 = self.activation_6(outs_branch2)
        outs_branch2 = self.res2b_branch2c(outs_branch2)
        outs_branch2 = self.bn2b_branch2c(outs_branch2)

        outs = tf.math.add_n((outs_branch2, outs_branch1))

        outs = self.activation_7(outs)

        outs_branch1 = outs
        outs_branch2 = self.res2c_branch2a(outs)
        outs_branch2 = self.bn2c_branch2a(outs_branch2)
        outs_branch2 = self.activation_8(outs_branch2)
        outs_branch2 = self.res2c_branch2b(outs_branch2)
        outs_branch2 = self.bn2c_branch2b(outs_branch2)
        outs_branch2 = self.activation_9(outs_branch2)
        outs_branch2 = self.res2c_branch2c(outs_branch2)
        outs_branch2 = self.bn2c_branch2c(outs_branch2)

        outs = tf.math.add_n((outs_branch2, outs_branch1))

        outs = self.activation_10(outs)

        return outs

class Resnet(keras.layers.Layer):
    def __init__(self):
        super(Resnet, self).__init__()
        self.block1 = Block(64, 256, is_maxpooling=True)
        self.block2 = Block(128, 512)
        self.block3 = Block(128, 1024)
        self.block4 = Block(512, 2048)

    def call(self, inputs, training=True):
        # (Batch, 128, 128, 256)
        inputs_1_4 = self.block1(inputs)
        # (Batch, 64, 64, 512)
        inputs_1_8 = self.block2(inputs_1_4)
        # (Batch, 32, 32, 1024)
        inputs_1_16 = self.block3(inputs_1_8)
        # (Batch, 16, 16, 2048)
        inputs_1_32 = self.block4(inputs_1_16)
        return [inputs_1_4, inputs_1_8, inputs_1_16, inputs_1_32]