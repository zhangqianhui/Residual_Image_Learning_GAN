import tensorflow as tf
from ops import batch_normal, conv2d, lrelu, de_conv
from utils import save_images
import numpy as np
import time


class reImageGAN(object):
    # build model
    def __init__(self, batch_size, max_iters, model_path, data_ob, sample_size, sample_path, log_dir, learning_rate,
                 data_format):

        self.batch_size = batch_size
        self.max_iters = max_iters
        self.pixel_model_path = model_path
        self.data_ob = data_ob
        self.sample_size = sample_size
        self.sample_path = sample_path
        self.log_dir = log_dir
        self.learning_rate = learning_rate
        self.log_vars = []
        self.channel = data_ob.channel
        self.shape = data_ob.shape
        self.class_nums = 2
        self.extra_class = 1
        self.data_format = data_format
        self.output_size = data_ob.image_size

        self.dom_1_label = tf.placeholder(tf.int32, [batch_size, 1])
        self.dom_2_label = tf.placeholder(tf.int32, [batch_size, 1])
        self.dom_1_images = tf.placeholder(tf.float32, [batch_size, self.output_size, self.output_size, self.channel])
        self.dom_2_images = tf.placeholder(tf.float32, [batch_size, self.output_size, self.output_size, self.channel])

    def build_model_reImageGAN(self):

        # Get the result of manipulating

        self.residual_img_1 = self.encode_decode_1(self.dom_1_images, False)
        self.residual_img_2 = self.encode_decode_2(self.dom_2_images, False)

        self.x_tilde_1 = self.residual_img_1 + self.dom_1_images
        self.x_tilde_2 = self.residual_img_2 + self.dom_2_images

        # the gan loss and classification loss
        self.dom_1_real_class_logits, self.dom_1_real_feature_mapping = self.discriminate(self.dom_1_images, False)
        self.dom_2_real_class_logits, self.dom_2_real_feature_mapping = self.discriminate(self.dom_2_images, True)
        self.dom_1_fake_class_logits, self.dom_1_fake_feature_mapping = self.discriminate(self.x_tilde_1, True)
        self.dom_2_fake_class_logits, self.dom_2_fake_feature_mapping = self.discriminate(self.x_tilde_2, True)

        self.dom_1_label = tf.squeeze(self.dom_1_label)
        self.dom_2_label = tf.squeeze(self.dom_2_label)

        self.dom_1_real_class_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.dom_1_real_class_logits,
            labels=tf.one_hot(tf.ones_like(self.dom_1_label), self.class_nums + self.extra_class,
                              dtype=tf.float32))

        self.dom_2_real_class_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.dom_2_real_class_logits,
            labels=tf.one_hot(tf.ones_like(self.dom_2_label) + 1, self.class_nums + self.extra_class,
                              dtype=tf.float32))

        self.dom_1_fake_class_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.dom_1_fake_class_logits,
            labels=tf.one_hot(tf.zeros_like(self.dom_1_label),
                              self.class_nums + self.extra_class,
                              dtype=tf.float32))

        self.dom_2_fake_class_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.dom_2_fake_class_logits,
            labels=tf.one_hot(tf.zeros_like(self.dom_1_label),
                              self.class_nums + self.extra_class,
                              dtype=tf.float32))

        self.D_loss = tf.reduce_mean(self.dom_1_real_class_cross_entropy + self.dom_1_fake_class_cross_entropy \
                                     + self.dom_2_real_class_cross_entropy + self.dom_2_fake_class_cross_entropy)

        # G_1 loss
        self.G_1_gan_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.dom_1_fake_class_logits,
            labels=tf.one_hot(tf.ones_like(self.dom_1_label) + 1,
                              self.class_nums + self.extra_class,
                              dtype=tf.float32)))

        self.G_1_resi_regu_loss = 0.00005 * tf.reduce_mean(tf.reduce_sum(tf.abs(self.residual_img_1), axis=[1, 2, 3]))

        self.G_1_feature_mapping_loss = 0.000005 * tf.reduce_mean(tf.reduce_sum(
            tf.abs(self.dom_1_real_feature_mapping - self.dom_1_fake_feature_mapping), axis=[1, 2, 3]))

        self.G_1_loss = self.G_1_gan_loss + self.G_1_resi_regu_loss + self.G_1_feature_mapping_loss

        # G_2 loss
        self.G_2_gan_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.dom_2_fake_class_logits,
                                                                                   labels=tf.one_hot(
                                                                                       tf.ones_like(self.dom_1_label),
                                                                                       self.class_nums + self.extra_class,
                                                                                       dtype=tf.float32)))

        self.G_2_resi_regu_loss = 0.00005 * tf.reduce_mean(tf.reduce_sum(tf.abs(self.residual_img_2), axis=[1, 2, 3]))

        self.G_2_feature_mapping_loss = 0.000005 * tf.reduce_mean(tf.reduce_sum(
            tf.abs(self.dom_2_real_feature_mapping - self.dom_2_fake_feature_mapping), axis=[1, 2, 3]))

        self.G_2_loss = self.G_2_gan_loss + self.G_2_resi_regu_loss + self.G_2_feature_mapping_loss

        self.log_vars.append(("D_loss", self.D_loss))
        self.log_vars.append(("G_1_loss", self.G_1_loss))
        self.log_vars.append(("G_2_loss", self.G_2_loss))

        self.t_vars = tf.trainable_variables()

        self.d_vars = [var for var in self.t_vars if 'discriminator' in var.name]
        self.g_1_vars = [var for var in self.t_vars if 'encode_decode_1' in var.name]
        self.g_2_vars = [var for var in self.t_vars if 'encode_decode_2' in var.name]

        print "d_vars", len(self.d_vars)
        print "g_1_vars", len(self.g_1_vars)
        print "g_2_vars", len(self.g_2_vars)

        self.saver = tf.train.Saver()

        for k, v in self.log_vars:
            tf.summary.scalar(k, v)

    # do train
    def train(self):

        d_trainer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9)
        d_gradients = d_trainer.compute_gradients(self.D_loss, var_list=self.d_vars)
        d_clipped_gradients = [(tf.clip_by_value(_[0], -1., 1.), _[1]) for _ in d_gradients]
        opti_D = d_trainer.apply_gradients(d_clipped_gradients)

        g_1_trainer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9)
        g_1_gradients = g_1_trainer.compute_gradients(self.G_1_loss, var_list=self.g_1_vars)
        g_1_gradients = [(tf.clip_by_value(_[0], -1., 1.), _[1]) for _ in g_1_gradients]
        opti_G1 = g_1_trainer.apply_gradients(g_1_gradients)

        g_2_trainer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9)
        g_2_gradients = g_2_trainer.compute_gradients(self.G_2_loss, var_list=self.g_2_vars)
        g_2_gradients = [(tf.clip_by_value(_[0], -1., 1.), _[1]) for _ in g_2_gradients]
        opti_G2 = g_2_trainer.apply_gradients(g_2_gradients)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)
            step = 0

            print "Starting the training"
            while step <= self.max_iters:

                # optimization D
                dom_1_train_data_list, dom_1_label_list, dom_2_train_data_list, dom_2_label_list = self.data_ob.getNextBatch(
                    step, self.batch_size)
                dom_1_batch_images_array = self.data_ob.getShapeForData(dom_1_train_data_list)
                dom_2_batch_images_array = self.data_ob.getShapeForData(dom_2_train_data_list)
                start_time = time.time()

                f_d = {self.dom_1_images: dom_1_batch_images_array, self.dom_2_images: dom_2_batch_images_array,
                       self.dom_1_label: dom_1_label_list, self.dom_2_label: dom_2_label_list}

                f_d_1 = {self.dom_1_images: dom_1_batch_images_array, self.dom_1_label: dom_1_label_list,
                         self.dom_2_label: dom_2_label_list}
                f_d_2 = {self.dom_2_images: dom_2_batch_images_array, self.dom_1_label: dom_1_label_list,
                         self.dom_2_label: dom_2_label_list}

                sess.run(opti_D, feed_dict=f_d)
                sess.run(opti_G1, feed_dict=f_d_1)
                sess.run(opti_G2, feed_dict=f_d_2)

                end_time = time.time() - start_time
                summary_str = sess.run(summary_op, feed_dict=f_d)
                summary_writer.add_summary(summary_str, step)

                if step % 50 == 0:
                    d_loss, G_1_loss, G_1_resi_regu_loss, G_1_gan_loss, G_1_fm_loss, G_2_loss, G_2_resi_regu_loss, G_2_fm_loss, G_2_gan_loss \
                        = sess.run([self.D_loss, self.G_1_loss, self.G_1_resi_regu_loss, self.G_1_gan_loss,
                                    self.G_1_feature_mapping_loss \
                                       , self.G_2_loss, self.G_2_resi_regu_loss, self.G_2_gan_loss,
                                    self.G_2_feature_mapping_loss], feed_dict=f_d)
                    print(
                    "step %d D_loss = %.7f, G_1_loss=%.7f, G_1_regu_loss=%.7f, G_1_gan_loss=%.7f G_1_fm_loss=%.7f, G_2_loss=%.7f,"
                    " G_2_regu_loss=%.7f, G_2_gan_loss=%.7f, G_2_fm_loss=%.7f , Time=%.3f" % (
                        step, d_loss, G_1_loss, G_1_resi_regu_loss, G_1_gan_loss, G_1_fm_loss,
                        G_2_loss, G_2_resi_regu_loss, G_2_gan_loss, G_2_fm_loss, end_time))

                if np.mod(step, 200) == 0 and step != 0:
                    save_images(dom_1_batch_images_array[0:64], [8, 8],
                                '{}/{:02d}_real_dom1.png'.format(self.sample_path, step))
                    save_images(dom_2_batch_images_array[0:64], [8, 8],
                                '{}/{:02d}_real_dom2.png'.format(self.sample_path, step))

                    r1, x_tilde_1, r2, x_tilde_2 = sess.run([self.residual_img_1, self.x_tilde_1,
                                                             self.residual_img_2, self.x_tilde_2], feed_dict=f_d)
                    save_images(r1[0:64], [8, 8], '{}/{:02d}_r1.png'.format(self.sample_path, step))
                    save_images(x_tilde_1[0:64], [8, 8], '{}/{:02d}_x_tilde1.png'.format(self.sample_path, step))

                    save_images(r2[0:64], [8, 8], '{}/{:02d}_r2.png'.format(self.sample_path, step))
                    save_images(x_tilde_2[0:64], [8, 8], '{}/{:02d}_x_tilde2.png'.format(self.sample_path, step))

                    self.saver.save(sess, self.pixel_model_path)

                step += 1

            save_path = self.saver.save(sess, self.pixel_model_path)
            print "Model saved in file: %s" % save_path

    def discriminate(self, x_var, reuse=False):

        with tf.variable_scope("discriminator") as scope:
            if reuse == True:
                scope.reuse_variables()

            conv1 = lrelu(conv2d(x_var, output_dim=64, name='dis_conv1'))
            conv2 = lrelu(batch_normal(conv2d(conv1, output_dim=256, name='dis_conv2'), scope='dis_bn1', reuse=reuse))
            conv3 = lrelu(batch_normal(conv2d(conv2, output_dim=512, name='dis_conv3'), scope='dis_bn2', reuse=reuse))
            conv4 = conv2d(conv3, output_dim=1024, name='dis_conv4')
            middle_feature = conv4
            conv4 = lrelu(batch_normal(conv4, scope='dis_bn3', reuse=reuse))
            conv5 = conv2d(conv4, output_dim=3, name='dis_conv5', padding='VALID')

            print conv5.shape

            return conv5, middle_feature

    def encode_decode_1(self, x, reuse=False):

        with tf.variable_scope("encode_decode_1") as scope:
            if reuse == True:
                scope.reuse_variables()

            conv1 = lrelu(batch_normal(conv2d(x, output_dim=64, name='e_c1'), scope='e_in1', reuse=reuse))
            conv2 = lrelu(batch_normal(conv2d(conv1, output_dim=128, name='e_c2'), scope='e_in2', reuse=reuse))
            conv3 = lrelu(batch_normal(conv2d(conv2, output_dim=512, name='e_c3'), scope='e_in3', reuse=reuse))

            # for x_{1}
            de_conv1 = lrelu(batch_normal(de_conv(conv3, output_shape=[self.batch_size, 16, 16, 128]
                                                  , name='e_d1'), scope='e_in4', reuse=reuse))
            de_conv2 = lrelu(batch_normal(de_conv(de_conv1, output_shape=[self.batch_size, 32, 32, 64]
                                                  , name='e_d2'), scope='e_in5', reuse=reuse))
            x_tilde1 = de_conv(de_conv2, output_shape=[self.batch_size] + self.shape, name='e_d3')

            return tf.tanh(x_tilde1)

    def encode_decode_2(self, x, reuse=False):

        with tf.variable_scope("encode_decode_2") as scope:
            if reuse == True:
                scope.reuse_variables()

            conv1 = lrelu(batch_normal(conv2d(x, output_dim=64, name='e_c1'), scope='e_in1', reuse=reuse))
            conv2 = lrelu(batch_normal(conv2d(conv1, output_dim=128, name='e_c2'), scope='e_in2', reuse=reuse))
            conv3 = lrelu(batch_normal(conv2d(conv2, output_dim=256, name='e_c3'), scope='e_in3', reuse=reuse))
            # for x_{1}
            de_conv1 = lrelu(batch_normal(de_conv(conv3, output_shape=[self.batch_size, 16, 16, 128]
                                                  , name='e_d1'), scope='e_in4', reuse=reuse))
            de_conv2 = lrelu(batch_normal(de_conv(de_conv1, output_shape=[self.batch_size, 32, 32, 64]
                                                  , name='e_d2'), scope='e_in5', reuse=reuse))
            x_tilde1 = de_conv(de_conv2, output_shape=[self.batch_size] + self.shape, name='e_d3')

            return tf.tanh(x_tilde1)



