import glob
from config import path_tfrecord
import os

def read_txt_file(file_path):
    '''
    读取文本文件数据
    file_path: 文件路径
    '''
    lines = []
    with open(file_path, 'r') as f:
        line = f.readline()
        while line is not None and line != '':
            line = line.strip('\ufeff').strip('\n')
            lines.append(line)
            line = f.readline()

    return lines

def get_image_len():
    '''
    获取图片数量
    '''
    imgs = glob.glob(os.path.join(TRAIN_CONFIG['dir_img'], '*.jpg'))
    length = 0
    try:
        length = len(imgs)
        return length
    except Exception as ex:
        raise ex
