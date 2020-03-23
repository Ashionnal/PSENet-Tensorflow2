from config import path_tfrecord, dir_gt, dir_img
import tensorflow as tf
import glob
import os
import cv2
from utils import file_util
import numpy as np

def icdar15_to_tfrecords(output_file, data_dir, gt_dir):
    '''
    icdar15数据转换成tfreocrds文件格式
    output_file: 输出文件
    data_dir: 数据文件路径
    gt_dir: 真值文件路径
    '''
    writer = tf.io.TFRecordWriter(output_file)
    images_path = glob.glob(os.path.join(data_dir, '*.jpg'))
    print('{0}\timages found in {1}'.format(len(images_path), data_dir))

    for idx, image_path in enumerate(images_path):
        '''
        oriented_boxes: 检测框数据, shape=(line_num, 8), 注意: 点的数据已被归一化
        axs: x的坐标, shape = (line_num, 4)
        ays: y的坐标, shape = (line_num, 4)
        num_list: 对于icdar15数据的点数量为4, shape = (line_num, 1)
        bboxes: 左上角坐标和右下角坐标, shape = (line_num, 4)
        labels: label的值，如果是###则为0，无需识别，否则为1
        labels_text: 对应真label内容，可用于ocr识别
        '''
        oriented_boxes = []
        axs, ays = [], []
        num_list = []
        bboxes = []
        labels = []
        labels_text = []

        image_full_name = image_path.split('/')[-1]
        image_name = image_full_name.split('.')[0]
        gt_name = 'gt_{0}.txt'.format(image_name)
        gt_path = os.path.join(gt_dir, gt_name)

        print('\tconverting image: {0}, \ttotal_legth: {1}, \tcurrent_idx: {2}'.format(image_name, len(images_path), idx + 1))

        im_data = cv2.imread(image_path)
        shape = im_data.shape
        h, w = shape[0:2]
        h *= 1.0
        w *= 1.0
        
        lines = file_util.read_txt_file(gt_path)
        for line in lines:
            gt = line.split(',')
            oriented_box = [int(gt[i]) for i in range(8)]
            oriented_box = np.asarray(oriented_box) / ([w, h] * 4)
            oriented_boxes.append(oriented_box)

            xs = np.reshape(oriented_box, (4, 2))[:, 0]
            ys = np.reshape(oriented_box, (4, 2))[:, 1]
            xmin = xs.min()
            xmax = xs.max()
            ymin = ys.min()
            ymax = ys.max()
            bboxes.append(np.array([xmin, ymin, xmax, ymax]))
            axs.append(xs)
            ays.append(ys)
            num_list.append(4)

            labels_text.append(str.encode(gt[-1]))
            ignored = '###' in gt[-1]
            if ignored:
                labels.append(0)
            else:
                labels.append(1)

        serialized_example = convert_to_example([im_data.tobytes()], 
                                                list(shape),
                                                np.asarray(axs),
                                                np.asarray(ays),
                                                num_list,
                                                np.asarray(bboxes),
                                                labels_text,
                                                labels)
        writer.write(serialized_example)
    writer.close()


def convert_to_example(im_data, shape, x, y, num, bboxes, word, label):
    '''
    序列化数据成tfrecords example
    具体参数含义请参考调用地方的注释
    '''
    feature = {}
    # 图片原始数据,未被归一化
    feature['image'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=im_data))
    # 图片shape
    feature['shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=shape))
    # x的坐标 shape = （num_box, 4）
    feature['xs'] = tf.train.Feature(float_list=tf.train.FloatList(value=x.flatten()))
    # y的坐标 shape = （num_box, 4）
    feature['ys'] = tf.train.Feature(float_list=tf.train.FloatList(value=y.flatten()))
    # 点的数量 shape = （num_box,） 值都为4
    feature['num_points'] = tf.train.Feature(int64_list=tf.train.Int64List(value=num))
    # 将原始的4个点的坐标做成左上角和右下角坐标形式 shape = (num_box, 4) 4->(xmin, ymin, xmax, ymax)
    feature['bboxes'] = tf.train.Feature(float_list=tf.train.FloatList(value=bboxes.flatten()))
    # 标签值的内容 shape = (num_box,)
    feature['words'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=word))
    # 如果标签值非###则为1，否则为0 shape = (num_box,)
    feature['labels'] = tf.train.Feature(int64_list=tf.train.Int64List(value=label))

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    serialized_example = example.SerializeToString()
    return serialized_example

if __name__ == "__main__":
    icdar15_to_tfrecords('/home/yanshuxuan/gitprojects/PSENet-Tensorflow2-New/dataset/icdar2015_temp_trian.tfreocrd',
                         dir_img,
                         dir_gt)