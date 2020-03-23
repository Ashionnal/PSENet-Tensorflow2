import tensorflow as tf
from tensorflow import keras
from core.nets.resnset import Resnet
from config import kernal_rate


class PSENet(keras.Model):
    def __init__(self):
        super(PSENet, self).__init__()
        self.resnet = Resnet()
        # p5层定义
        self.p5_conv2d = keras.layers.Conv2D(filters=256, kernel_size=1, padding='same')
        self.p5_bn = keras.layers.BatchNormalization()
        self.p5_activation = keras.layers.ReLU()
        self.p5_upsample2d = keras.layers.UpSampling2D(8, interpolation='bilinear')
        # p4层定义
        self.p4_conv2d = keras.layers.Conv2D(filters=256, kernel_size=1, padding='same')
        self.p4_bn = keras.layers.BatchNormalization()
        self.p4_activation = keras.layers.ReLU()
        self.p4_upsample2d = keras.layers.UpSampling2D(2, interpolation='bilinear')
        self.p4_upsample2d_forconcat = keras.layers.UpSampling2D(4, interpolation='bilinear')
        # p3层定义
        self.p3_conv2d = keras.layers.Conv2D(filters=256, kernel_size=1, padding='same')
        self.p3_bn = keras.layers.BatchNormalization()
        self.p3_activation = keras.layers.ReLU()
        self.p3_upsample2d = keras.layers.UpSampling2D(2, interpolation='bilinear')
        self.p3_upsample2d_forconcat = keras.layers.UpSampling2D(2, interpolation='bilinear')
        # p2层定义
        self.p2_conv2d = keras.layers.Conv2D(filters=256, kernel_size=1, padding='same')
        self.p2_bn = keras.layers.BatchNormalization()
        self.p2_activation = keras.layers.ReLU()
        self.p2_upsample2d = keras.layers.UpSampling2D(2, interpolation='bilinear')
        self.p2_upsample2d_forconcat = keras.layers.UpSampling2D(1, interpolation='bilinear')
        # f_multi层定义
        self.f_multi_conv2d = keras.layers.Conv2D(filters=256, kernel_size=3, padding='same')
        self.f_multi_bn = keras.layers.BatchNormalization()
        self.f_multi_activation = keras.layers.ReLU()
        # seg_map
        self.seg_map_upsample2d = keras.layers.UpSampling2D(4, interpolation='bilinear')

    def __feature_fusion(self, features):
        '''
        特征融合
        features[0]   ->  p2: (Batch, 128, 128, 256)
        features[1]   ->  p3: (Batch, 64, 64, 512)
        features[2]   ->  p4: (Batch, 32, 32, 1024)
        features[3]   ->  p5: (Batch, 16, 16, 2048)
        '''
        p = [None] * 4
        # 进行p5层的特征上采样
        p[3] = self.p5_conv2d(features[3])
        p[3] = self.p5_bn(p[3])
        p[3] = self.p5_activation(p[3])
        F_multi = self.p5_upsample2d(p[3])

        # 进行p4层的特征上采样和特征融合
        p[2] = self.p4_conv2d(features[2])
        p[2] = self.p4_bn(p[2])
        p[2] = self.p4_activation(p[2])
        p[2] = tf.add(self.p4_upsample2d(p[3]), p[2])
        F_multi = tf.concat((F_multi, self.p4_upsample2d_forconcat(p[2])), axis=-1)

        # 进行p3层的特征上采样和特征融合
        p[1] = self.p3_conv2d(features[1])
        p[1] = self.p3_bn(p[1])
        p[1] = self.p3_activation(p[1])
        p[1] = tf.add(self.p3_upsample2d(p[2]), p[1])
        F_multi = tf.concat((F_multi, self.p3_upsample2d_forconcat(p[1])), axis=-1)

        # 进行p2层的特征上采样和特征融合
        p[0] = self.p2_conv2d(features[0])
        p[0] = self.p2_bn(p[0])
        p[0] = self.p2_activation(p[0])
        p[0] = tf.add(self.p2_upsample2d(p[1]), p[0])
        F_multi = tf.concat((F_multi, self.p2_upsample2d_forconcat(p[0])), axis=-1)

        F_multi = self.f_multi_conv2d(F_multi)
        F_multi = self.f_multi_bn(F_multi)
        # 最终融合的特征feature
        feature = self.f_multi_activation(F_multi)

        for i in range(len(kernal_rate) + 1):
            seg_map = keras.layers.Conv2D(filters=1, kernel_size=1)(feature)
            seg_map = self.seg_map_upsample2d(seg_map)
            # seg_map = tf.image.resize(seg_map, (tf.shape(seg_map)[1] * int(4), tf.shape(seg_map)[2] * int(4)))
            seg_map = tf.math.sigmoid(seg_map)
            if i == 0:
                seg_maps = seg_map
            else:
                seg_maps = tf.concat((seg_maps, seg_map), axis=-1)

        return seg_maps

    def call(self, inputs, istraing=True):
        '''
        返回seg_maps: 特征融合的数据，shape = (batch_size, top_feature_size, top_feature_size, len(kernal_rate) + 1) 
                      针对当前的参数 shape = (batch_size, 512, 512, 2)
        '''
        features = self.resnet(inputs)
        seg_maps = self.__feature_fusion(features)
        return seg_maps, features


if __name__ == "__main__":
    import cv2
    import numpy as np
    im_data = cv2.imread('/home/yanshuxuan/gitprojects/data/icdar_2015/ch4_training_images/img_1.jpg')
    im_data = np.resize(im_data, (512, 512, 3))
    im_data = np.expand_dims(im_data, 0).astype(np.float32)
    psenet = PSENet()
    psenet(im_data)
