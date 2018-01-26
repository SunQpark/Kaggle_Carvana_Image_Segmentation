import tensorflow as tf
from keras.utils import to_categorical
from model import UNET_MODEL


class TRAIN:
    def __init__(self, sess=None, weight=False):
        self.x = tf.placeholder(dtype='float32', shape=[None, self.length, self.length, 1], name='input_image')
        self.y = tf.placeholder(dtype='float32', shape=[None, self.length, self.length, 2], name='input_label')
        if sess is not None:
            self.sess = sess
        self.weight = weight

    def train(self, epoch, redundancy, keep_prob):  # 200 per training(0~199), 8 per validation(200-207)

        sess = self.sess
        phase = tf.placeholder(tf.bool)
        hypothesis = self.build_model(self.x, keep_prob=keep_prob, phase_train=phase)

        with tf.name_scope("loss"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=self.y))

        optimize = tf.train.AdamOptimizer(learning_rate=5e-4).minimize(loss)
        """
        with tf.name_scope("dev_acc"):
            correct_prediction = tf.equal(tf.argmax(hypothesis, 3), tf.argmax(self.y, 3))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        """
        training_epoch = epoch
        batch_size = 2  # each file has 155 photographs
        batch_photo_size = batch_size * (155 - 2 * redundancy)
        total_number = 140 * (155 - 2 * redundancy)
        num_batch = int(total_number / batch_photo_size)

        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        if self.weight:
            saver.restore(sess, './saved_model/my_final_model')
        writer = tf.summary.FileWriter('./board/sum')
        writer.add_graph(sess.graph)

        for i in range(training_epoch):
            for j in range(num_batch):
                INPUT_FILE(batch_size, j * batch_size, redundancy, self.edge_x, self.edge_y, self.length)
                with open('/srv/repository/data/image.txt', 'rb') as f_image, open('/srv/repository/data/label.txt',
                                                                                   'rb') as f_label:
                    batch_image = pickle.load(f_image)
                    batch_label = pickle.load(f_label)

                    sess.run(optimize, feed_dict={self.x: batch_image, self.y: batch_label, phase: True})
                    cost = sess.run(loss, feed_dict={self.x: batch_image, self.y: batch_label, phase: True})
                    print("epoch : ", i + 1, ", batch : ", j + 1, ", loss : ", cost)

                if (j + 1) % 10 == 0:
                    # acc = sess.run(accuracy, feed_dict={self.x: batch_image, self.y: batch_label, phase: True})
                    # print('accuracy is :', acc)
                    saver.save(sess, './saved_model/my_final_model')
                    print('model saved')


    def test(self, redundancy):  # total number of files in training : 2 per each sequence(208~209)
        sess = self.sess
        phase = tf.placeholder(tf.bool)
        prediction = self.build_model(self.x, keep_prob=1.0, phase_train=phase)

        saver = tf.train.Saver()
        num_test_file = 1

        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, './saved_model/my_final_model')
        with open('./dice.txt', 'w') as Store_Dice:
            for i in range(num_test_file):
                # initialize
                INPUT_FILE(1, i + 205, redundancy, self.edge_x, self.edge_y, self.length)
                temp_TP = temp_FP = temp_FN = 0
                with open('/srv/repository/data/image.txt', 'rb') as t_image, open('/srv/repository/data/label.txt',
                                                                                   'rb') as t_label:
                    batch_image = pickle.load(t_image)
                    batch_label = pickle.load(t_label)
                    pred_image = sess.run(prediction, feed_dict={self.x: batch_image, phase: False})
                    with open('./matplot_array.txt', 'wb') as mat_array:
                        pickle.dump(pred_image, mat_array)
                    with open('./matplot_label.txt', 'wb') as mat_label:
                        pickle.dump(batch_label, mat_label)
                    dice = self.dice_test(output=prediction, target=self.y, axis=[0, 1, 2, 3])
                    output_dice = sess.run(dice, feed_dict={self.x: batch_image, self.y: batch_label, phase: False})

                    # Store_Dice.write('dice : %f \n' % output_dice)
                    print('dice : ', output_dice)

                    # Store_Dice.write('total dice : %f ' % output_dice)

        return dice