import tensorflow as tf
from config import min_size, image_random_scale, max_shrink, kernal_rate, r_g_b_means
import cv2
import numpy as np
import Polygon as plg
import pyclipper
import random
from PIL import Image


def tf_image_whitened(image, means=r_g_b_means):
    '''
    调整图片数据，减去各个通道均值
    image: 图片数据
    means: 需要减去的均值
    '''
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    mean = tf.constant(means, dtype=image.dtype)
    image = image - mean
    return image

def random_crop(imgs, img_size=min_size):
    '''
    图片，真值，训练任务数据, kernal数据 裁剪
    imgs: (图片数据，真值数据，训练任务数据，kernal数据)
    img_size: 裁剪之后的图片信息数据
    '''
    h, w = imgs[0].shape[0:2]
    th, tw = img_size, img_size
    if h < th or w < tw:
        for idx in range(len(imgs)):
            image = imgs[idx]
            color = [123., 117., 104.] if len(image.shape) == 3 else [0]
            top = (th - h) // 2 if (th -h) > 0 else 0
            bottom = th - top - h if (th - h) > 0else 0
            left = (tw - w) // 2 if (tw - w) > 0 else 0

            imgs[idx] = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    h, w = imgs[0].shape[0:2]
    if w == tw and h == th:
        return imgs
    
    # 前面判断不清楚, 估计是随机使用，有百分之37.5的概率将需要处理，后面的判断是判断gt_text中存在标记为正
    if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
        # tl(top_left): 最左上角有文字区域标记的点
        tl = np.min(np.where(imgs[1] > 0), axis=1) - img_size
        tl[tl < 0] = 0
        # br(bottom_right): 最右下角有文字区域标记的点
        br = np.max(np.where(imgs[1] > 0), axis=1) - img_size
        br[br < 0] = 0

        br[0] = min(br[0], h - th)
        br[1] = min(br[1], w - tw)

        i = random.randint(tl[0], br[0])
        j = random.randint(tl[1], br[1])
    else:
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            imgs[idx] = imgs[idx][i:i + th, j:j + tw, :]
        else:
            imgs[idx] = imgs[idx][i:i + th, j:j + tw]
    return imgs

def random_rotate(imgs):
    '''
    图片，真值，训练任务数据，kernal数据 旋转
    imgs: (图片数据，真值数据，训练任务数据, kernal数据)
    '''
    max_angle = 10
    # 控制旋转角度在 -10 ～ 10之间
    angle = random.random() * 2 * max_angle - max_angle
    for i in range(len(imgs)):
        img = imgs[i]
        h, w = img.shape[:2]
        # 计算旋转矩阵getRotationMatrix2D(中心点坐标, 旋转角度, 1代表等比例缩放)
        rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        # warpAffine 获取仿射后的图像
        img_rotation = cv2.warpAffine(img, rotation_matrix, (w, h))
        imgs[i] = img_rotation
    return imgs

def random_hrizontal_flip(imgs):
    '''
    图片，真值，训练任务，kernal数据垂直翻转
    imgs: (图片数据，真值数据，训练任务数据, kernal)
    '''
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs

def dist(a, b):
    '''
    计算两个点之间的距离，使用三角公式
    a: a点坐标, shape = (2,)，即(x1, y1)
    b: b点坐标, shape = (2,)，即(x2, y2)
    '''
    return np.sqrt(np.sum((a - b) ** 2))

def perimeter(bbox):
    '''
    计算每一个检测框的周长
    bbox: 单个检测框点的坐标，(4, 2)，
    '''
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri

def shrink(bboxes, rate, max_shr=max_shrink):
    '''
    此方法是参照 https://zhuanlan.zhihu.com/p/91019893 里面的方法
    具体可以参考官方论文的公式 di的计算
    bboxes: 单张图片的所有bboxes数据 shape = (num_boxes, 8)
    max_shr: 最大向里收缩的轮廓di距离
    '''
    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        # 计算面积
        area = plg.Polygon(bbox).area()
        # 计算周长
        peri = perimeter(bbox)
        # 下面两行是生成轮廓使用
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        # 计算di也就是偏移距离，这是根据公式计算
        offset = min((int)(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)
        # 开始生成向内偏移offset轮廓
        shrinked_bbox = pco.Execute(-offset)
        # 如果生成的轮廓没有或者轮廓数据非法则直接用原始的bbox数据
        if len(shrinked_bbox) == 0:
            shrinked_bboxes.append(bbox)
            continue
        shrinked_bbox = np.array(shrinked_bbox[0])
        if shrinked_bbox.shape[0] <= 2:
            shrinked_bboxes.append(bbox)
            continue
        shrinked_bboxes.append(shrinked_bbox)
    return np.array(shrinked_bboxes)

def random_scale(img, min_size=min_size, ran_scale=image_random_scale):
    '''
    图片随机缩放处理
    img: 图片原始数据
    min_size: 输入到网络中的图片尺寸大小
    ran_scale: 图片缩放比例因子
    '''
    h, w = img.shape[0:2]
    if max(h, w) > 1280:
        scale = 1280. / max(h, w)
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    h, w = img.shape[0:2]
    random_scale = np.array(ran_scale)
    scale = np.random.choice(random_scale)
    if min(h, w) * scale <= min_size:
        scale = (min_size + 10) * 1.0 / min(h, w)
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img

def process_data_np(image, label, bboxes):
    '''
    处理图片数据
    image: 原始图片数据
    label: 图片标签数据
    bboxes: 即原来的polys，包含了4个点坐标信息，shape = (num_boxes, 8)
    '''
    img = random_scale(image)
    # 此时的img数据已被处理成min_size的形状
    gt_text = np.zeros(img.shape[0:2], dtype=np.uint8)
    training_mask = np.ones(img.shape[0:2], dtype=np.uint8)
    if bboxes.shape[0] > 0:
        # 因img未被归一化，但是bboxes的数据之前是被归一化后的，所以需要将bboxes的点坐标数据转换成当前图片尺寸大小的坐标信息
        bboxes_reshape_data = bboxes * ([img.shape[1], img.shape[0]] * 4)
        # 原始bboxes形状是shape = (num_boxes, 8)，将其转化成(num_boxes, 4, 2)的形式
        bboxes_reshape_shape = (bboxes.shape[0], int(bboxes.shape[1] / 2), 2)
        bboxes = np.reshape(bboxes_reshape_data, bboxes_reshape_shape).astype(np.int32)
        # 循环当前图片里所有的bboxes数据
        for i in range(bboxes.shape[0]):
            # 将bboxes[i]（第i个检测框）所在的点轮廓信息在 training_mask中标记为 ,i + 1
            # 画出特征图值，进行标记，注意的cv2的图片是(h, w, 3)
            cv2.drawContours(gt_text, [bboxes[i]], -1, i + 1, -1)
            # 可以查看下图中效果，将i + 1改成255白色像素值，可以看的更加明显
            # cv2.imshow('123', gt_text)
            # cv2.waitKey(0)
            if not label[i]:
                # training_mask默认是都需要训练，对于label标记为0（即###）的则赋值成0，不需要进行训练
                cv2.drawContours(training_mask, [bboxes[i]], -1, 0, -1)

    gt_kernals = []
    for rate in kernal_rate:
        gt_kernal = np.zeros(img.shape[0:2], dtype=np.uint8)
        kernal_bboxes = shrink(bboxes, rate)
        for i in range(bboxes.shape[0]):
            # 将gt_kernal对应的轮廓数据标记为1
            cv2.drawContours(gt_kernal, [kernal_bboxes[i]], -1, 1, -1)
        gt_kernals.append(gt_kernal)

    imgs = [img, gt_text, training_mask]
    # 注意，这边为什么没直接 ims=[img, gt_text, training_mask], 因为gt_kernal的shape为(num,)
    imgs.extend(gt_kernals)
    # 图片等数据垂直翻转
    imgs = random_hrizontal_flip(imgs)
    # 图片等数据旋转
    imgs = random_rotate(imgs)
    # 图片等数据裁剪
    imgs = random_crop(imgs)
    img, gt_text, training_mask, gt_kernals = imgs[0], imgs[1], imgs[2], imgs[3:]
    # 因为上面不同的检测框的轮廓的标记是从1往上递增的，所以这边直接将非0的真值标记为1
    gt_text[gt_text > 0] = 1
    gt_kernals = np.array(gt_kernals)
    # 将array转成img形式，数据格式就是为uint8
    img = Image.fromarray(img)
    img = np.asarray(img)
    return img, gt_text, gt_kernals, training_mask

def process_data(image, label, polys, num_points, bboxes):
    '''
    image: 原始图片数据 shape = (720, 1980)
    label: 标签数据，(num_boxes,)
    polys：4个坐标点, shape = (num_boxes, 8) 8 -> (x0, y0, x1, y1, x2, y2, x3, y3)
    num_points: (num_boxes,) 里面的值都是4
    bboxes: 左上角右下角坐标 shape = (num_boxes, 4) 4 -> (xmin, ymin, xmax, ymax)
    '''
    img, gt_text, gt_kernals, training_mask = tf.numpy_function(process_data_np, 
                                                                [image, label, polys], 
                                                                [tf.uint8, tf.uint8, tf.uint8, tf.uint8])
    img.set_shape((min_size, min_size, 3))
    gt_text.set_shape((min_size, min_size))
    gt_kernals.set_shape((len(kernal_rate), min_size, min_size))
    training_mask.set_shape((min_size, min_size))

    img = tf.cast(img, tf.float32)
    gt_text = tf.cast(gt_text, tf.float32)
    gt_kernals = tf.cast(gt_kernals, tf.float32)
    training_mask = tf.cast(training_mask, tf.float32)

    # 调整亮度
    img = tf.image.random_brightness(img, max_delta=32./255.)
    # 调整饱和度
    img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
    img = tf_image_whitened(img)
    
    return (img, gt_text, gt_kernals, training_mask)