import tensorflow as tf
from utils import mkdir_p
from utils import CelebA
from ReImageGAN import reImageGAN

flags = tf.app.flags
flags.DEFINE_integer("OPER_FLAG", 0, "the flag of opertion")
flags.DEFINE_string("OPER_NAME", "RIGAN_new3_regu_fp", "the name of opertion")
flags.DEFINE_string("IMAGE_PATH", "/home/haha/data/celebA", "the path of your celebA")
FLAGS = flags.FLAGS

if __name__ == "__main__":

    root_log_dir = "./ganceleba_log/logs{}".format(FLAGS.OPER_FLAG)
    semigan_checkpoint_dir = "./model_gan{}/model.ckpt".format(FLAGS.OPER_NAME)
    sample_path = "./ganceleba_sample/sample{}/sample_{}".format(FLAGS.OPER_FLAG, FLAGS.OPER_NAME)

    mkdir_p(root_log_dir)
    mkdir_p(semigan_checkpoint_dir)
    mkdir_p(sample_path)
    model_path = semigan_checkpoint_dir

    batch_size = 64
    max_iters = 40000
    sample_size = 128
    learn_rate = 0.0002

    OPER_FLAG = FLAGS.OPER_FLAG
    data_format = 'NHWC'

    m_ob = CelebA(FLAGS.IMAGE_PATH)

    print "dom1_train_data_list", len(m_ob.dom_1_train_data_list)
    print "dom2_train_data_list", len(m_ob.dom_2_train_data_list)
    print "the number of train data", len(m_ob.dom_1_train_data_list + m_ob.dom_2_train_data_list)

    reGAN = reImageGAN(batch_size=batch_size, max_iters=max_iters,
                      model_path= model_path, data_ob=m_ob, sample_size= sample_size,
                      sample_path =sample_path , log_dir= root_log_dir , learning_rate= learn_rate, data_format=data_format)

    if OPER_FLAG == 0:

        reGAN.build_model_reImageGAN()
        reGAN.train()


