import tensorflow as tf
from ops import conv2d, lrelu, de_conv, log_sum_exp, instance_norm
from utils import save_images
import numpy as np
import time

class reImageGAN(object):

    # build model
    def __init__(self, batch_size, max_iters, model_path, data_ob, sample_path, log_dir, learning_rate, l1_w, per_w):

        self.batch_size = batch_size
        self.max_iters = max_iters
        self.pixel_model_path = model_path
        self.data_ob = data_ob
        self.sample_path = sample_path
        self.log_dir = log_dir
        self.learning_rate = learning_rate
        self.log_vars = []
        self.channel = data_ob.channel
        self.shape = data_ob.shape
        self.class_nums = 2
        self.output_size = data_ob.image_size
        self.l1_w = l1_w
        self.per_w = per_w

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

        self.dom_1_real_class_logits = tf.squeeze(self.dom_1_real_class_logits)
        self.dom_2_real_class_logits = tf.squeeze(self.dom_2_real_class_logits)
        self.dom_1_fake_class_logits = tf.squeeze(self.dom_1_fake_class_logits)
        self.dom_2_fake_class_logits = tf.squeeze(self.dom_2_fake_class_logits)

        self.dom_1_real_class_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.dom_1_real_class_logits,
            labels=tf.one_hot(tf.zeros_like(self.dom_1_label), self.class_nums,
                              dtype=tf.int32)))

        self.dom_2_real_class_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.dom_2_real_class_logits,
            labels=tf.one_hot(tf.ones_like(self.dom_2_label), self.class_nums, dtype=tf.int32)))

        self.D_gan_loss_1 = - 0.5 * tf.reduce_mean(log_sum_exp(self.dom_1_real_class_logits)) + 0.5 * tf.reduce_mean(tf.nn.softplus(log_sum_exp(self.dom_1_real_class_logits))) \
                   + 0.5 * tf.reduce_mean(tf.nn.softplus(log_sum_exp(self.dom_2_fake_class_logits)))
        self.D_gan_loss_2 = - 0.5 * tf.reduce_mean(log_sum_exp(self.dom_2_real_class_logits)) + 0.5 * tf.reduce_mean(tf.nn.softplus(log_sum_exp(self.dom_2_real_class_logits))) \
                   + 0.5 * tf.reduce_mean(tf.nn.softplus(log_sum_exp(self.dom_1_fake_class_logits)))

        self.D_loss = self.dom_1_real_class_cross_entropy + self.dom_2_real_class_cross_entropy \
                      + self.D_gan_loss_1 + self.D_gan_loss_2

        # G_1 loss
        self.G_1_class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.dom_1_fake_class_logits,
            labels=tf.one_hot(tf.ones_like(self.dom_1_label),
                              self.class_nums,
                              dtype=tf.float32)))

        self.G_1_gan_loss = -0.5 * tf.reduce_mean(tf.reduce_logsumexp(self.dom_1_fake_class_logits, 1)) \
                      + 0.5 * tf.reduce_mean(tf.nn.softplus(tf.reduce_logsumexp(self.dom_1_fake_class_logits, 1)))

        self.G_1_resi_regu_loss = self.l1_w * tf.reduce_mean(tf.reduce_sum(tf.abs(self.residual_img_1), axis=[1,2,3]))

        self.G_1_feature_mapping_loss = self.per_w * tf.reduce_mean(tf.reduce_sum(
            tf.abs(self.dom_1_real_feature_mapping - self.dom_1_fake_feature_mapping), axis=[1,2,3]))

        self.G_1_loss = self.G_1_class_loss + self.G_1_resi_regu_loss + self.G_1_feature_mapping_loss + self.G_1_gan_loss

        # G_2 loss
        self.G_2_class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.dom_2_fake_class_logits,
                                                                                   labels=tf.one_hot(
                                                                                       tf.zeros_like(self.dom_1_label),
                                                                                       self.class_nums,
                                                                                       dtype=tf.float32)))

        self.G_2_gan_loss = -0.5 * tf.reduce_mean(tf.reduce_logsumexp(self.dom_2_fake_class_logits, 1)) \
                      + 0.5 * tf.reduce_mean(tf.nn.softplus(tf.reduce_logsumexp(self.dom_2_fake_class_logits, 1)))

        self.G_2_resi_regu_loss = self.l1_w * tf.reduce_mean(tf.reduce_sum(tf.abs(self.residual_img_2), axis=[1,2,3]))
        self.G_2_feature_mapping_loss = self.per_w * tf.reduce_mean(tf.reduce_sum(
            tf.abs(self.dom_2_real_feature_mapping - self.dom_2_fake_feature_mapping), axis=[1,2,3]))

        self.G_2_loss = self.G_2_class_loss + self.G_2_resi_regu_loss + self.G_2_feature_mapping_loss + self.G_2_gan_loss

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

        opti_D = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9).minimize(self.D_loss, var_list=self.d_vars)
        opti_G1 = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9).minimize(self.G_1_loss, var_list=self.g_1_vars)
        opti_G2 = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9).minimize(self.G_2_loss, var_list=self.g_2_vars)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            # self.saver.restore(sess, self.pixel_model_path)
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

                    d_loss, G_1_loss, G_1_resi_regu_loss, G_1_fm_loss, G_2_loss, G_2_resi_regu_loss, G_2_fm_loss\
                        = sess.run([self.D_loss, self.G_1_loss, self.G_1_resi_regu_loss,
                                    self.G_1_feature_mapping_loss \
                                       , self.G_2_loss, self.G_2_resi_regu_loss,
                                    self.G_2_feature_mapping_loss], feed_dict=f_d)
                    print(
                    "step %d D_loss = %.7f, G_1_loss=%.7f, G_1_regu_loss=%.7f,  G_1_fm_loss=%.7f, G_2_loss=%.7f,"
                    " G_2_regu_loss=%.7f, G_2_fm_loss=%.7f , Time=%.3f" % (
                        step, d_loss, G_1_loss, G_1_resi_regu_loss, G_1_fm_loss, G_2_loss, G_2_resi_regu_loss, G_2_fm_loss, end_time))

                if np.mod(step, 200) == 0 and step != 0:

                    save_images(dom_1_batch_images_array[0:self.batch_size], [self.batch_size/8, 8],
                                '{}/{:02d}_real_dom1.png'.format(self.sample_path, step))
                    save_images(dom_2_batch_images_array[0:self.batch_size], [self.batch_size/8, 8],
                                '{}/{:02d}_real_dom2.png'.format(self.sample_path, step))

                    r1, x_tilde_1, r2, x_tilde_2 = sess.run([self.residual_img_1, self.x_tilde_1,
                                                             self.residual_img_2, self.x_tilde_2], feed_dict=f_d)

                    x_tilde_1 = np.clip(x_tilde_1, -1, 1)
                    x_tilde_2 = np.clip(x_tilde_2, -1, 1)

                    save_images(r1[0:self.batch_size], [self.batch_size/8, 8], '{}/{:02d}_r1.png'.format(self.sample_path, step))
                    save_images(x_tilde_1[0:64], [self.batch_size/8, 8], '{}/{:02d}_x_tilde1.png'.format(self.sample_path, step))

                    save_images(r2[0:self.batch_size], [self.batch_size/8, 8], '{}/{:02d}_r2.png'.format(self.sample_path, step))
                    save_images(x_tilde_2[0:self.batch_size], [self.batch_size/8, 8], '{}/{:02d}_x_tilde2.png'.format(self.sample_path, step))

                    self.saver.save(sess, self.pixel_model_path)

                step += 1

            save_path = self.saver.save(sess, self.pixel_model_path)
            print "Model saved in file: %s" % save_path

    def discriminate(self, x_var, reuse=False):

        with tf.variable_scope("discriminator") as scope:
            if reuse == True:
                scope.reuse_variables()

            conv1 = lrelu(conv2d(x_var, output_dim=64, name='dis_conv1'))
            conv2 = lrelu(instance_norm(conv2d(conv1, output_dim=128, name='dis_conv2'), scope='dis_bn1'))
            conv3 = lrelu(instance_norm(conv2d(conv2, output_dim=256, name='dis_conv3'), scope='dis_bn2'))
            conv4 = conv2d(conv3, output_dim=512, name='dis_conv4')
            middle_conv = conv4
            conv4 = lrelu(instance_norm(conv4, scope='dis_bn3'))
            conv5 = lrelu(instance_norm(conv2d(conv4, output_dim=1024, name='dis_conv5'), scope='dis_bn4'))

            conv6 = conv2d(conv5, output_dim=2, k_w=4, k_h=4, d_h=1, d_w=1, padding='VALID', name='dis_conv6')

            return conv6, middle_conv

    def encode_decode_1(self, x, reuse=False):

        with tf.variable_scope("encode_decode_1") as scope:
            if reuse == True:
                scope.reuse_variables()

            conv1 = lrelu(instance_norm(conv2d(x, output_dim=64, k_w=5, k_h=5, d_w=1, d_h=1, name='e_c1'), scope='e_in1'))
            conv2 = lrelu(instance_norm(conv2d(conv1, output_dim=128, name='e_c2'), scope='e_in2'))
            conv3 = lrelu(instance_norm(conv2d(conv2, output_dim=256, name='e_c3'), scope='e_in3'))
            # for x_{1}
            de_conv1 = lrelu(instance_norm(de_conv(conv3, output_shape=[self.batch_size, 64, 64, 128]
                                                  , name='e_d1', k_h=3, k_w=3), scope='e_in4'))
            de_conv2 = lrelu(instance_norm(de_conv(de_conv1, output_shape=[self.batch_size, 128, 128, 64]
                                                  , name='e_d2', k_w=3, k_h=3), scope='e_in5'))
            x_tilde1 = conv2d(de_conv2, output_dim=3, d_h=1, d_w=1, name='e_c4')

            return x_tilde1

    def encode_decode_2(self, x, reuse=False):

        with tf.variable_scope("encode_decode_2") as scope:
            if reuse == True:
                scope.reuse_variables()

            conv1 = lrelu(instance_norm(conv2d(x, output_dim=64, k_w=5, k_h=5, d_w=1, d_h=1, name='e_c1'), scope='e_in1',
                                       ))
            conv2 = lrelu(instance_norm(conv2d(conv1, output_dim=128, name='e_c2'), scope='e_in2'))

            conv3 = lrelu(instance_norm(conv2d(conv2, output_dim=256, name='e_c3'), scope='e_in3'))
            # for x_{1}
            de_conv1 = lrelu(instance_norm(de_conv(conv3, output_shape=[self.batch_size, 64, 64, 128]
                                                  , name='e_d1', k_h=3, k_w=3), scope='e_in4',
                                          ))
            de_conv2 = lrelu(instance_norm(de_conv(de_conv1, output_shape=[self.batch_size, 128, 128, 64]
                                                  , name='e_d2', k_w=3, k_h=3), scope='e_in5',
                                          ))
            x_tilde1 = conv2d(de_conv2, output_dim=3, d_h=1, d_w=1, name='e_c4')

            return x_tilde1





