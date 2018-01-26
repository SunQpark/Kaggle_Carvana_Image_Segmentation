import tensorflow as tf
from model import UNET_MODEL

with tf.Session() as sess:
    BT_model = UNET_MODEL(sess=sess, edge_x=60, edge_y=40, length=160)
    BT_model.train(epoch=10, keep_prob=1.0, redundancy=15)
    #acc, dice = BT_model.test()
    #print(acc, dice)