import tensorflow as tf
from train import TRAIN

with tf.Session() as sess:
    a = TRAIN(sess=sess)
    a.train(10, 1)
