import tensorflow as tf


class UNET_MODEL:
    def build_model(self, image, keep_prob, phase_train):  # phase_train = tf.bool- train or test
        shape = tf.shape(image)
        # Conv1_1, 1_2 & Pool1
        filt1_1, _, bias1_1 = self.get_weight_bias(3, 64)
        conv1_1 = self.conv_layer(image, filt1_1, bias1_1, name='conv1_1')
        relu1_1 = tf.nn.relu(conv1_1, name='relu1_1')
        filt1_2, _, bias1_2 = self.get_weight_bias(64, 64)
        conv1_2 = self.conv_layer(relu1_1, filt1_2, bias1_2, name='conv1_2')
        relu1_2 = tf.nn.relu(conv1_2, name='relu1_2')
        skip9 = relu1_2[:, 88:480, 88:480, :]
        pool1 = self.maxpool_layer(relu1_2, name='pool1')

        # Conv2_1, 2_2 & Pool2
        filt2_1, _, bias2_1 = self.get_weight_bias(64, 128)
        conv2_1 = self.conv_layer(pool1, filt2_1, bias2_1, name='conv2_1')
        relu2_1 = tf.nn.relu(conv2_1, name='relu2_1')
        filt2_2, _, bias2_2 = self.get_weight_bias(128, 128)
        conv2_2 = self.conv_layer(relu2_1, filt2_2, bias2_2, name='conv2_2')
        relu2_2 = tf.nn.relu(conv2_2, name='relu2_2')
        skip8 = relu2_2[:, 40:240, 40:240, :]
        pool2 = self.maxpool_layer(relu2_2, name='pool2')

        # Conv3_1, 3_2 & Pool3
        filt3_1, _, bias3_1 = self.get_weight_bias(128, 256)
        conv3_1 = self.conv_layer(pool2, filt3_1, bias3_1, name='conv3_1')
        relu3_1 = tf.nn.relu(conv3_1, name='relu3_1')
        filt3_2, _, bias3_2 = self.get_weight_bias(256, 256)
        conv3_2 = self.conv_layer(relu3_1, filt3_2, bias3_2, name='conv3_2')
        relu3_2 = tf.nn.relu(conv3_2, name='relu3_2')
        skip7 = relu3_2[:, 16:120, 16:120, :]
        pool3 = self.maxpool_layer(relu3_2, name='pool3')

        # Conv4_1, 4_2 & Pool4
        filt4_1, _, bias4_1 = self.get_weight_bias(256, 512)
        conv4_1 = self.conv_layer(pool3, filt4_1, bias4_1, name='conv4_1')
        relu4_1 = tf.nn.relu(conv4_1, name='relu4_1')
        filt4_2, _, bias4_2 = self.get_weight_bias(512, 512)
        conv4_2 = self.conv_layer(relu4_1, filt4_2, bias4_2, name='conv4_2')
        relu4_2 = tf.nn.relu(conv4_2, name='relu4_2')
        skip6 = relu4_2[:, 4:60, 4:60, :]
        pool4 = self.maxpool_layer(relu4_2, name='pool4')

        # Conv5_1, 5_2 & Pool5
        filt5_1, _, bias5_1 = self.get_weight_bias(512, 1024)
        conv5_1 = self.conv_layer(pool4, filt5_1, bias5_1, name='conv5_1')
        relu5_1 = tf.nn.relu(conv5_1, name='relu5_1')
        filt5_2, _, bias5_2 = self.get_weight_bias(1024, 1024)
        conv5_2 = self.conv_layer(relu5_1, filt5_2, bias5_2, name='conv5_2')
        relu5_2 = tf.nn.relu(conv5_2, name='relu4_2')

        # Deconv6, Conv6_1, 6_2
        filt6, bias6, _ = self.get_weight_bias(512, 1024)
        deconv6 = self.deconv_layer(relu5_2, filt6, bias6, [shape[0], 56, 56, 512], name='deconv6')
        relu6 = tf.nn.relu(deconv6, name='relu6')
        concat6 = tf.concat([skip6, relu6], axis=3, name='concat6')
        filt6_1, _, bias6_1 = self.get_weight_bias(1024, 512)
        conv6_1 = self.conv_layer(concat6, filt6_1, bias6_1, name='conv6_1')
        relu6_1 = tf.nn.relu(conv6_1, name='relu6_1')
        filt6_2, _, bias6_2 = self.get_weight_bias(512, 512)
        conv6_2 = self.conv_layer(relu6_1, filt6_2, bias6_2, name='conv6_2')
        relu6_2 = tf.nn.relu(conv6_2, name='relu6_2')

        # Deconv7, Conv7_1, 7_2
        filt7, bias7, _ = self.get_weight_bias(256, 512)
        deconv7 = self.deconv_layer(relu6_2, filt7, bias7, [shape[0], 104, 104, 256], name='deconv7')
        relu7 = tf.nn.relu(deconv7, name='relu7')
        concat7 = tf.concat([skip7, relu7], axis=3, name='concat7')
        filt7_1, _, bias7_1 = self.get_weight_bias(512, 256)
        conv7_1 = self.conv_layer(concat7, filt7_1, bias7_1, name='conv7_1')
        relu7_1 = tf.nn.relu(conv7_1, name='relu7_1')
        filt7_2, _, bias7_2 = self.get_weight_bias(256, 256)
        conv7_2 = self.conv_layer(relu7_1, filt7_2, bias7_2, name='conv7_2')
        relu7_2 = tf.nn.relu(conv7_2, name='relu7_2')

        # Deconv8, Conv8_1, 8_2
        filt8, bias8, _ = self.get_weight_bias(128, 256)
        deconv8 = self.deconv_layer(relu7_2, filt8, bias8, [shape[0], 200, 200, 128], name='deconv8')
        relu8 = tf.nn.relu(deconv8, name='relu8')
        concat8 = tf.concat([skip8, relu8], axis=3, name='concat8')
        filt8_1, _, bias8_1 = self.get_weight_bias(256, 128)
        conv8_1 = self.conv_layer(concat8, filt8_1, bias8_1, name='conv8_1')
        relu8_1 = tf.nn.relu(conv8_1, name='relu8_1')
        filt8_2, _, bias8_2 = self.get_weight_bias(128, 128)
        conv8_2 = self.conv_layer(relu8_1, filt8_2, bias8_2, name='conv8_2')
        relu8_2 = tf.nn.relu(conv8_2, name='relu8_2')


        # Deconv9, Conv9_1, 9_2
        filt9, bias9, _ = self.get_weight_bias(64, 128)
        deconv9 = self.deconv_layer(relu8_2, filt9, bias9, [shape[0], 392, 392, 64], name='deconv9')
        relu9 = tf.nn.relu(deconv9, name='relu9')
        concat9 = tf.concat([skip9, relu9], axis=3, name='concat9')
        filt9_1, _, bias9_1 = self.get_weight_bias(128, 64)
        conv9_1 = self.conv_layer(concat9, filt9_1, bias9_1, name='conv9_1')
        relu9_1 = tf.nn.relu(conv9_1, name='relu9_1')
        filt9_2, _, bias9_2 = self.get_weight_bias(64, 64)
        conv9_2 = self.conv_layer(relu9_1, filt9_2, bias9_2, name='conv9_2')
        relu9_2 = tf.nn.relu(conv9_2, name='relu9_2')

        filt10 = tf.Variable(tf.truncated_normal([1, 1, 64, 1], dtype='float32', stddev=1e-1), name='filter')
        bias10 = tf.Variable(tf.constant(0, shape=[1], dtype='float32'), name='bias')
        conv10 = self.conv_layer(relu9_2, filt10, bias10, name='conv10')

        return conv10

    def deconv_layer(self, input, filter, bias, shape, name):
        with tf.variable_scope(name):
            deconv = tf.nn.conv2d_transpose(input, filter, strides=[1, 2, 2, 1], output_shape=shape)
            deconv = tf.add(deconv, bias)
        return deconv

    def conv_layer(self, input, filter, bias, name):
        with tf.variable_scope(name):
            conv = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
            conv = tf.nn.bias_add(conv, bias)
        return conv

    def maxpool_layer(self, input, name):
        return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def get_weight_bias(self, shape1, shape2):
        weight = tf.Variable(tf.truncated_normal([3, 3, shape1, shape2], dtype='float32', stddev=1e-1), name='filter')
        bias1 = tf.Variable(tf.constant(0, shape=[shape1], dtype='float32'), name='bias')
        bias2 = tf.Variable(tf.constant(0, shape=[shape2], dtype='float32'), name='bias')# not necessary because of batch_norm
        return weight, bias1, bias2

    def batch_norm(self, input, channel_length, phase_train):  # input's 4th dimension = channel length
        with tf.variable_scope('batch_norm'):
            beta = tf.Variable(tf.constant(0.0, shape=[channel_length]),
                               name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[channel_length]),
                                name='gamma', trainable=True)
            batch_mean, batch_var = tf.nn.moments(input, [0, 1, 2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.9)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(phase_train, mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(input, mean, var, beta, gamma, 1e-3)

            activation = tf.nn.relu(normed, name='relu')
        return activation

    def drop_out(self, shape, keep_prob, name):
        return tf.nn.dropout(shape, keep_prob=keep_prob, name=name)