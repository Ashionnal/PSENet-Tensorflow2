
# tfrecord文件保存文件夹路径
path_tfrecord = '/home/yanshuxuan/gitprojects/PSENet-Tensorflow2/dataset/icdar2015_trian.tfreocrd'
# 图片文件夹路径
dir_img = '/home/yanshuxuan/gitprojects/data/icdar_2015/ch4_training_images'
# 真值文件夹路径
dir_gt = '/home/yanshuxuan/gitprojects/data/icdar_2015/ch4_training_localization_transcription_gt'
# 缩小比例，即核的数量
kernal_rate = [0.5, 0.75]
# 训练次数
epochs = 1000
# 批次大小
batch_size = 4
# 每一个检测框所需的点的数量，这边是固定
points_count = 4

#
min_size = 512
# 图片缩放比例
image_random_scale = [0.5, 1.0, 1.5, 2.0]
# 最大向里收缩的轮廓di距离
max_shrink = 20
r_g_b_means = [123., 117., 104.]
# 是否需要ohem数据增强
ohem = True
# Lc Loss的比例
lc_loss_rate = 0.7
# l2 loss权重
weight_decay = 1e-5
# 日志文件夹
log_dir = '/home/yanshuxuan/gitprojects/PSENet-Tensorflow2/logs'
# 模型文件夹
model_dir = '/home/yanshuxuan/gitprojects/PSENet-Tensorflow2/model'
# 是否加载预训练模型
pre_train = True
# tflite 路径
model_tflite_path = '/home/yanshuxuan/gitprojects/PSENet-Tensorflow2/model_tflite'
# 测试图片文件夹路径
test_image_dir = '/home/yanshuxuan/gitprojects/PSENet-Tensorflow2/images'

threshold = 0.55
