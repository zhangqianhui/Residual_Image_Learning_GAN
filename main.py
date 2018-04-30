import tensorflow as tf
from utils import mkdir_p
from utils import CelebA
from ResidualGAN import reImageGAN

flags = tf.app.flags
flags.DEFINE_integer("OPER_FLAG", 0, "Train or Test")
flags.DEFINE_string("OPER_NAME", "1_6_2", "the name of opertionby the current time")
flags.DEFINE_string("IMAGE_PATH", "/home/?/data/celebA/", "the path of your celebA, don't contain other sub-directory")
flags.DEFINE_integer("BATCH_SIZE", 16, "the batch_size")
flags.DEFINE_integer("IMG_SIZE", 128, "the size of training samples")
flags.DEFINE_integer("MAX_ITERS", 50000, "the maxization of iterations")
flags.DEFINE_integer("LEARN_RATE", 0.0001, "the learning rate")
flags.DEFINE_integer("L1_W", 0.00005, "weight of L1 norm")
flags.DEFINE_integer("PER_W", 0.000005, "weight of perception loss")
flags.DEFINE_integer("recon_w", 10, "weight of recon loss")
flags.DEFINE_integer("attri_id", 20, "the id of attribute defined in the list_attr_celeba.txt")
FLAGS = flags.FLAGS

if __name__ == "__main__":

    root_log_dir = "./output/logs/logs{}".format(FLAGS.OPER_FLAG)
    gan_checkpoint_dir = "./output/model_gan/{}_model.ckpt".format(FLAGS.OPER_FLAG)
    sample_path = "./output/sample{}/sample_{}".format(FLAGS.OPER_NAME, FLAGS.OPER_FLAG)

    mkdir_p(root_log_dir)
    mkdir_p(gan_checkpoint_dir)
    mkdir_p(sample_path)
    model_path = gan_checkpoint_dir

    m_ob = CelebA(FLAGS.IMAGE_PATH, FLAGS.IMG_SIZE, FLAGS.attri_id)

    print "dom1_train_data_list", len(m_ob.dom_1_train_data_list)
    print "dom2_train_data_list", len(m_ob.dom_2_train_data_list)
    print "the number of train data", len(m_ob.dom_1_train_data_list + m_ob.dom_2_train_data_list)

    reGAN = reImageGAN(batch_size= FLAGS.BATCH_SIZE, max_iters= FLAGS.MAX_ITERS,
                      model_path= model_path, data_ob= m_ob,
                      sample_path= sample_path , log_dir= root_log_dir ,
                       learning_rate= FLAGS.LEARN_RATE, l1_w= FLAGS.L1_W, per_w= FLAGS.PER_W, recon_w= FLAGS.recon_w)

    if FLAGS.OPER_FLAG == 0:

        reGAN.build_model_reImageGAN()
        reGAN.train()

