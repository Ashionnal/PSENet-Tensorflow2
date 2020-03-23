
import glob
from config import kernal_rate, threshold, test_image_dir, min_size
import tensorflow as tf
import numpy as np
from skimage.measure import label
import os
import cv2
from PSE_C import mylib

def expansion(CC, Si):
    def check(arr):
        if arr.shape[-1] == 1:
            arr = np.squeeze(arr, -1)
            return arr.astype(np.int32)
        else:
            return arr.astype(np.int32)
    CC = check(CC)
    Si = check(Si)

    CC_out = CC.copy(order='C')
    ps = mylib.PyExpand()
    ps.expansion(CC_out, Si.copy(order='C'))
    return CC_out

def process_map(segment_map):
    segment_map = [np.squeeze(seg, 0) for seg in segment_map]
    S1 = (segment_map[-1]) > threshold
    CC = label(S1, connectivity=2)
    expand_cc = CC
    for i in range(len(segment_map) - 2, -1, -1):
        S_i = segment_map[i] > threshold
        expand_cc = expansion(expand_cc, S_i)

    return expand_cc

def rect_to_xys(rect, image_shape):
    h, w = image_shape[0:2]
    def get_valid_x(x):
        if x < 0:
            return 0
        if x >= w:
            return w - 1
        return x

    def get_valid_y(y):
        if y < 0:
            return 0
        if y >= h:
            return h - 1
        return y

    points = cv2.boxPoints(rect)
    points = np.int0(points)
    for i_xy, (x, y) in enumerate(points):
        x = get_valid_x(x)
        y = get_valid_y(y)
        points[i_xy, :] = [x, y]
    points = np.reshape(points, -1)
    return points

def region_to_bbox(mask, image_size, min_height=10, min_area=300):
    h, w = image_size
    score_map = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(score_map.copy(), mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(contours[0])
    rect = list(rect)
    sw, sh = rect[1][:]
    rect = tuple(rect)
    xys = rect_to_xys(rect, [h, w])
    return xys

def map_to_bboxes(segment_maps, result_map, image_size, aver_score=0.9):
    cc_num = result_map.max()
    bboxes = np.empty((0, 8))
    scores = np.empty((0, 1))
    if cc_num <= 0:
        return None, None
    for i in range(1, cc_num + 1):
        mask = (result_map == i)
        region_score = np.sum(mask * np.squeeze(segment_maps[0], (0, -1))) / np.sum(mask)
        if region_score > aver_score:
            bbox = region_to_bbox(mask, image_size)
        else:
            bbox = None
        if bbox is not None:
            bboxes = np.concatenate((bboxes, bbox[np.newaxis, :]), axis=0)
            scores = np.concatenate((scores, np.array([[region_score]])), 0)

    return bboxes, scores

def eval_model(psenet):
    image_size = (768, 1280)
    images = glob.glob(os.path.join(test_image_dir, '*.jpg'))
    for image in images:
        im_data = cv2.imread(image)
        image_source_shape = im_data.shape
        source_im_data = im_data
        im_data = tf.expand_dims(tf.image.resize(im_data, (min_size, min_size)), 0)
        seg_maps, _ = psenet(im_data, istraing=False)

        seg_map_list = []
        for i in range(len(kernal_rate) + 1):
            seg_map_list.append(tf.image.resize(seg_maps[:, :, :, i:i+1], [tf.shape(im_data)[0], tf.shape(im_data)[1]]))

        mask = tf.greater_equal(seg_map_list[0], threshold)
        mask = tf.cast(mask, dtype=tf.float32)

        seg_map_list = [seg_map * mask for seg_map in seg_map_list]
        result_map = process_map(seg_map_list)
        bboxes, scores = map_to_bboxes(seg_map_list, result_map, image_size)
        if bboxes is not None:
            for i, bbox in enumerate(bboxes):
                print('scores-------------', scores[i])
                print('bbox---------------', str(bbox))
                points = np.array([[int(bbox[0]), int(bbox[1])], [int(bbox[2]), int(bbox[3])], [int(bbox[4]), int(bbox[5])], [int(bbox[6]), int(bbox[7])]], np.int32)
                points = points.reshape((-1, 1, 2))
                cv2.polylines(source_im_data, [points], True, (0, 255, 255))
                cv2.imshow('123', source_im_data)
                cv2.waitKey(0)
        pass