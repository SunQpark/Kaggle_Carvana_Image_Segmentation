import tensorflow as tf
from keras.utils import to_categorical
from model import UNET_MODEL
from utils.patch_data import InputImage
import pickle as pkl
import gzip


class TRAIN:
    def __init__(self, sess=None, weight=False):
        self.x = tf.placeholder(dtype='float32', shape=[None, 572, 572, 3], name='input_image')
        self.y = tf.placeholder(dtype='float32', shape=[None, 388, 388, 1], name='input_label')
        if sess is not None:
            self.sess = sess
        self.weight = weight

    def train(self, epoch, keep_prob):

        sess = self.sess
        phase = tf.placeholder(tf.bool)
        unet = UNET_MODEL()
        hypothesis = unet.build_model(image=self.x, keep_prob=keep_prob, phase_train=phase)

        with tf.name_scope("loss"):
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=hypothesis, labels=self.y))

        optimize = tf.train.AdamOptimizer(learning_rate=5e-4).minimize(loss)
        """
        with tf.name_scope("dev_acc"):
            correct_prediction = tf.equal(tf.argmax(hypothesis, 3), tf.argmax(self.y, 3))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        """
        training_epoch = epoch
        batch_size = 1  # number of image per batch
        total_number = 1000  #should find exact number
        num_batch = int(total_number / batch_size)

        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        if self.weight:
            saver.restore(sess, './saved_model/my_final_model')
        writer = tf.summary.FileWriter('./board/sum')
        writer.add_graph(sess.graph)

        for i in range(training_epoch):
            for j in range(num_batch):
                InputImage(order=j, num_images_per_batch=batch_size)
                with gzip.open('inputs/data_patch/data.pkl.gz', 'rb') as f:
                    batch_image, batch_label = pkl.load(f)
                # sess.run(optimize, feed_dict={self.x: batch_image, self.y: batch_label, phase: True})
                _, cost = sess.run([optimize, loss], feed_dict={self.x: batch_image, self.y: batch_label, phase: True})
                print("epoch : ", i + 1, ", batch : ", j + 1, ", loss : ", cost)

                if (j + 1) % 1 == 0:
                    # acc = sess.run(accuracy, feed_dict={self.x: batch_image, self.y: batch_label, phase: True})
                    # print('accuracy is :', acc)
                    saver.save(sess, './saved_model/my_final_model')
                    print('model saved')