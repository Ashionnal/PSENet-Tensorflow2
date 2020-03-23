from config import epochs, batch_size, weight_decay, min_size, log_dir, model_dir, pre_train
from dataset.dataload import Dataload
from loss import PSENetLoss
from core.nets.psenet import PSENet
import tensorflow as tf
import time
import math
import os
from loss1 import calcute_loss
import datetime


dataload = Dataload()
dataset = dataload.dataset
loss = PSENetLoss()
psenet = PSENet()
optimizer = tf.keras.optimizers.Adam()
writer = tf.summary.create_file_writer(log_dir)
total_step =  math.ceil(dataload.total_count / batch_size) * epochs
step = 1

if pre_train:
    filepath = os.path.join(model_dir, 'model-31500')
    psenet.load_weights(filepath=filepath)
    print("Successfully load weights from {}!".format(filepath))

def print_model_summary(net):
    net.build(input_shape=(None, min_size, min_size, 3))
    net.summary()


def train_step(img, gt_text, gt_kernals, training_mask):
    with tf.GradientTape() as tape:
        preds, _  = psenet(img)
        losses, Lc_Loss, Ls_Loss = loss.loss(preds, gt_text, gt_kernals, training_mask)
        # losses = calcute_loss(preds, gt_text, gt_kernals, training_mask)

        variables = psenet.trainable_variables
        l2_loss = weight_decay * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in variables])
        total_loss = losses + l2_loss
    gradients = tape.gradient(total_loss, variables)
    optimizer.apply_gradients(grads_and_vars=zip(gradients, variables))

    tf.summary.scalar('loss', losses, step)
    tf.summary.scalar('Lc_Loss', Lc_Loss, step)
    tf.summary.scalar('Ls_Loss', Ls_Loss, step)
    tf.summary.scalar('total_loss', total_loss, step)

    print('total_step: {},step: {},total_loss:{:6f},l2_loss:{:6f},loss: {:6f},lc_loss: {:6f},ls_loss: {:6f}'.format(str(total_step).zfill(10), str(step).zfill(10), total_loss, l2_loss, losses, Lc_Loss, Ls_Loss))

print_model_summary(psenet)

if __name__ == "__main__":
    with writer.as_default():
        for epoch in range(epochs):
            for batch_data in dataset:
                img, gt_text, gt_kernals, training_mask = batch_data
                train_step(img, gt_text, gt_kernals, training_mask)
                writer.flush()
                step += 1

                if step % 500 == 0:
                    psenet.save_weights(filepath=os.path.join(model_dir, 'model-{}'.format(step)), save_format="tf")

    writer.close()
