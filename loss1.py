import tensorflow as tf
from config import kernal_rate, ohem, lc_loss_rate

def cal_dice_loss(pred, gt):
    '''
    计算LOSS值, 这边采用的公式是 D(Si, Gi) = 2 * ∑x,y(Si,x,y * Gi,x,y) / (∑x,y Si,x,y ^ 2 +  ∑x,y Gi,x,y ^ 2)
            Lc = 1 - D(Sn * M, Gn * M)
    pred: 预测值shape = (BATCH_SIZE, 512, 512)
    gt: 真值 shape = (BATCH_SIZE, 512, 512)
    '''
    union = tf.reduce_sum(tf.multiply(pred, gt), [1, 2])
    pred_square = tf.reduce_sum(tf.square(pred), [1, 2])
    gt_square = tf.reduce_sum(tf.square(gt), [1, 2])

    dice_loss = 1. - (2 * union + 1e-5) / (pred_square + gt_square + 1e-5)

    return dice_loss

def calcute_loss(pred_seg_maps, gt_map, kernals, training_mask, scope=None):
    '''
    计算loss，包括了 Lc为文本区域分类损失，Ls为收缩文本实例损失
                Lc = 1 - D(Sn * M, Gn * M)
                        1   n-1
                Ls = 1 - --- ( ∑ D(Si * W, Gi * W))
                        n-1  u=1
                W = 1 if Sn,x,y >= 0.5 else 0
    pred_seg_maps: 预测(BATCH, 512, 512, 4)
    gt_map: 标签数据 (BATCH, 512, 512)
    kernals: 内核数据 (BATCH, len(rate), 512, 512)
    training_mask: 训练标记 (BATCH, 512, 512)
    '''
    with tf.name_scope('Loss'):
        n = len(kernal_rate) + 1
        pred_text_map = pred_seg_maps[:, :, :, 0]
        # 对于预测的数据过滤掉不需要进行判断的检测区域并置为0，同时筛选出概率值大于0.5的数据，转换为float32
        mask = tf.cast(tf.greater(pred_text_map * training_mask, 0.5), tf.float32)
        # 对于预测的数据过滤掉不需要进行判断的检测区域并置为0
        pred_text_map = pred_text_map * training_mask
        # 对于真值数据过滤掉不需要进行判断的检测区域并置为0
        gt_map = gt_map * training_mask

        def online_hard_example_mining(maps):
            pred_map, gt_map = maps

            # 找到真值为1的正样本数据并赋值为True，否则其他为False（positive）shape=(BATCH_SIZE, 512, 512)
            pos_mask = tf.cast(tf.equal(gt_map, 1.), dtype=tf.float32)
            # 找到真值为0的负样本数据并赋值为True，否则其他为False（negtive）shape=(BATCH_SIZE, 512, 512)
            neg_mask = tf.cast(tf.equal(gt_map, 0.), dtype=tf.float32)
            n_pos = tf.reduce_sum((pos_mask), [0, 1])

            neg_val_all = tf.boolean_mask(pred_map, neg_mask)
            n_neg = tf.minimum(tf.shape(neg_val_all)[-1], tf.cast(n_pos * 3, tf.int32))
            n_neg = tf.cond(tf.greater(n_pos, 0), lambda: n_neg, lambda: tf.shape(neg_val_all)[-1])
            neg_hard, neg_idxs = tf.nn.top_k(neg_val_all, k=n_neg)
            neg_min = tf.cond(tf.greater(tf.shape(neg_hard)[-1], 0), lambda: neg_hard[-1], lambda:1.)

            neg_hard_mask = tf.cast(tf.greater_equal(pred_map, neg_min), dtype=tf.float32)
            pred_ohem = pos_mask * pred_map + neg_hard_mask * neg_mask * pred_map
            return pred_ohem, gt_map
        
        if ohem:
            pred_maps, gt_maps = tf.map_fn(online_hard_example_mining, (pred_text_map, gt_map))
        else:
            pred_maps, gt_maps = pred_text_map, gt_map

        ohem_dice_loss = cal_dice_loss(pred_maps, gt_maps)
        dice_loss = tf.reduce_mean(ohem_dice_loss)

        loss = []
        # tf.Graph().add_to_collection('losses', TRAIN_CONFIG['location_loss_rate'] * dice_loss)
        loss.append(lc_loss_rate * dice_loss)

        # 下面计算Ls文本实例收缩loss
        for i, _ in enumerate(kernal_rate):
            pred_map = pred_seg_maps[:, :, :, i + 1]
            gt_map = kernals[:, i, :, :]

            pred_map = pred_map * mask
            gt_map = gt_map * mask

            dice_loss = cal_dice_loss(pred_map, gt_map)
            dice_loss = tf.reduce_mean(dice_loss)
            # tf.Graph().add_to_collection('losses', (1 - 0.7) * dice_loss / (n - 1))
            loss.append((1 - 0.7) * dice_loss / (n - 1))
    # return tf.add_n(tf.Graph().get_collection('losses'), name='total_loss')
    return tf.add_n(loss, name='total_loss')