from config import path_tfrecord, batch_size, points_count
from dataset.preprocess import process_data
import tensorflow as tf

class Dataload():
    def __init__(self, is_training=True):
        self.tfreocrd_path = path_tfrecord
        self.is_training = is_training
        self.__build_data()

    def __process_data(self, im_data, label, polys, num_points, bboxes):
        return process_data(im_data, label, polys, num_points, bboxes)

    def __get_length(self, dataset):
        count = 0
        for _ in dataset:
            count += 1
        self.total_count = count

    def __build_data(self):
        print('load file: {0}'.format(path_tfrecord))
        raw_dataset = tf.data.TFRecordDataset(path_tfrecord)
        

        def __parse_serialize_example(serialized_example):
            '''
            解析序列化的tfrecord数据
            serialized_example: 原始的tfrecord example数据
            '''
            expected_to_features = {'image': tf.io.FixedLenFeature([], tf.string), 
                                    'shape': tf.io.FixedLenFeature([3], tf.int64),
                                    'xs': tf.io.VarLenFeature(tf.float32), 
                                    'ys': tf.io.VarLenFeature(tf.float32), 
                                    'num_points': tf.io.VarLenFeature(tf.int64), 
                                    'bboxes': tf.io.VarLenFeature(tf.float32), 
                                    'words': tf.io.VarLenFeature(tf.string),
                                    'labels': tf.io.VarLenFeature(tf.int64),
            }
            parsed_features = tf.io.parse_single_example(serialized_example, expected_to_features)
            im_data = tf.io.decode_raw(parsed_features['image'], tf.uint8)
            # im_data = tf.cast(im_data, dtype=tf.float64)
            im_data = tf.reshape(im_data, parsed_features['shape'])
            # 个人感觉这边的sparse_to_dense是没有必要的，因为这边的数据很完整
            num_points = tf.sparse.to_dense(parsed_features['num_points'], default_value=0)
            bboxes = tf.sparse.to_dense(parsed_features['bboxes'], default_value=0)
            bboxes = tf.reshape(bboxes, (-1, 4))
            x = tf.sparse.to_dense(parsed_features['xs'], default_value=0)
            xs = tf.reshape(x, (-1, points_count, 1))
            ys = tf.reshape(x, (-1, points_count, 1))

            # 先将原来的x坐标和y的坐标分别reshape成(1, 4, 1)
            # 然后将x和y的坐标最后一维度进行拼接，则会形成(1, 4, 2)
            # 意思就是N组数据，有4个点，每个点是2个值，分别是x和y的坐标
            polys = tf.concat((xs, ys), axis=-1)
            polys = tf.reshape(polys, (-1, points_count * 2))
            label = tf.sparse.to_dense(parsed_features['labels'], default_value=0)
            # im_data: 图片数据, shape = (720, 1280)
            # label: [0, 1, ..., 0] shape = (NUM,) 标签
            # polys: 点的坐标, shape = (NUM, 8)
            # num_points: 多少个点, 一般就是4
            # bboxes: bouding box坐标值，是左上角坐标和右下角坐标, shape = (NUM, 4)
            return im_data, label, polys, num_points, bboxes

        dataset = raw_dataset.map(__parse_serialize_example)
        self.__get_length(dataset)
        if self.is_training:
            dataset = dataset.map(self.__process_data)
            dataset = dataset.batch(batch_size)
        else:
            dataset.batch(1)
        self.dataset = dataset