import os.path
import time

import numpy as np
import tensorflow as tf
import cv2
from numba import jit
from PIL import Image

import bouncing_balls as b
import layer_def as ld
import BasicConvLSTMCell

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './checkpoints/re_lstm',
                           """dir to store trained net""")
tf.app.flags.DEFINE_integer('seq_length', 60,
                            """size of hidden layer""")
tf.app.flags.DEFINE_integer('seq_start', 30,
                            """ start of seq generation""")
tf.app.flags.DEFINE_integer('max_step', 301,
                            """max num of steps""")
tf.app.flags.DEFINE_float('keep_prob', .8,
                          """for dropout""")
tf.app.flags.DEFINE_float('lr', .001,
                          """for dropout""")
tf.app.flags.DEFINE_integer('batch_size', 50,
                            """batch size for training""")
tf.app.flags.DEFINE_float('weight_init', .1,
                          """weight init for fully connected layers""")


set_shape=64

@jit
def imgchange(img):
    temp=np.zeros((set_shape,set_shape,1))
    for i in range(set_shape):
        for j in range(set_shape):
            if img[i][j][0] == 255:
                temp[i][j][0] = 0.0
            else:
                temp[i][j][0] = float(img[i][j][0]) / 255
    return temp



@jit
def imgunchange(img):
    for i in range(set_shape):
        for j in range(set_shape):
            if img[i][j][0] == 0:
                img[i][j][0] = 255
    return img

def generate_bouncing_ball_sample(batch_size, seq_length, shape):
    dat = np.zeros((batch_size, seq_length, shape, shape,1))
    url='E:/dataset/test'

    dirs = os.listdir(url)
    i=0

    for dir in dirs:
        j = 0
        tmp_path = os.path.join(url, dir)
        files=os.listdir(tmp_path)

        for file in  files:
            tmp_file = os.path.join(tmp_path,file)

            temp_img = cv2.imread(tmp_file)
            re_img = cv2.resize(temp_img,(set_shape,set_shape))
            img=imgchange(re_img)
            dat[i, j, :, :,:]=img
            j=j+1
            if j==seq_length:
                break
        i=i+1
        if i==batch_size:
            break
    return dat


def network(inputs, hidden, lstm=True):
    # conv1
    conv1 = ld.conv_layer(inputs, 3, 2, 8, "encode_1")
    # conv2
    conv2 = ld.conv_layer(conv1, 3, 1, 8, "encode_2")
    # conv3
    conv3 = ld.conv_layer(conv2, 3, 2, 8, "encode_3")
    # conv4
    conv4 = ld.conv_layer(conv3, 3, 1, 8, "encode_4")
    # conv5
    conv5 = ld.conv_layer(conv4, 3, 2, 8, "encode_5")

    conv6 = ld.conv_layer(conv5, 1, 1, 4, "encode_6")
    y_0 = conv6

    with tf.variable_scope('conv_lstm', initializer=tf.random_uniform_initializer(-.01, 0.1)):
        cell = BasicConvLSTMCell.BasicConvLSTMCell([8, 8], [3, 3], 4)
        if hidden is None:
            hidden = cell.zero_state(FLAGS.batch_size, tf.float32)
        y_1, hidden = cell(y_0, hidden)


    # conv5
    conv5 = ld.transpose_conv_layer(y_1, 1, 1, 8, "decode_5")
    # conv6
    conv6 = ld.transpose_conv_layer(conv5, 3, 2, 8, "decode_6")
    # conv7
    conv7 = ld.transpose_conv_layer(conv6, 3, 1, 8, "decode_7")
    conv8 = ld.transpose_conv_layer(conv7, 3, 2, 8, "decode_8")
    conv9 = ld.transpose_conv_layer(conv8, 3, 1, 8, "decode_9")
    # x_1
    x_1 = ld.transpose_conv_layer(conv9, 3, 2, 1, "decode_10", True)  # set activation to linear

    return x_1, hidden


# make a template for reuse
network_template = tf.make_template('network', network)


def train():
    """Train ring_net for a number of steps."""
    with tf.Graph().as_default():
        # make inputs
        x = tf.placeholder(tf.float32, [None, FLAGS.seq_length, set_shape, set_shape, 1])

        # possible dropout inside
        keep_prob = tf.placeholder("float")
        x_dropout = tf.nn.dropout(x, keep_prob)

        # create network
        print('create network')
        x_unwrap = []

        # conv network
        hidden = None
        for i in range(FLAGS.seq_length-1):
            if i < FLAGS.seq_start:
                x_1, hidden = network_template(x_dropout[:, i, :, :, :], hidden)
            else:
                x_1, hidden = network_template(x_1, hidden)
            x_unwrap.append(x_1)

        # pack them all together
        x_unwrap = tf.stack(x_unwrap)
        x_unwrap = tf.transpose(x_unwrap, [1, 0, 2, 3, 4])

        # calc total loss (compare x_t to x_t+1)
        loss = tf.nn.l2_loss(x[:, FLAGS.seq_start + 1:, :, :, :] - x_unwrap[:, FLAGS.seq_start:, :, :, :])
        tf.summary.scalar('loss', loss)

        # training
        train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)

        # List of all Variables
        variables = tf.global_variables()

        # Build a saver
        saver = tf.train.Saver(tf.global_variables())
        # Summary op
        summary_op = tf.summary.merge_all()

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph.
        sess = tf.Session()

        # init if this is the very time training
        print("init network from scratch")
        sess.run(init)

        # Summary op
        graph_def = sess.graph.as_graph_def(add_shapes=True)
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph_def=graph_def)
        print('load data')
        dat = generate_bouncing_ball_sample(FLAGS.batch_size, FLAGS.seq_length, set_shape)
        print('start train')
        model_file = tf.train.latest_checkpoint(FLAGS.train_dir)
        saver.restore(sess, model_file)
        ims = sess.run([x_unwrap], feed_dict={x: dat, keep_prob: FLAGS.keep_prob})
        bitch=0
        ims = ims[0][bitch]
        print(ims.shape)
        print("now generating git!")
        images = []
        for i in range(FLAGS.seq_length - FLAGS.seq_start):
            x_1_r = np.uint8(np.maximum(ims[i, :, :, :], 0) * 255)
            x1=imgunchange(x_1_r)
            # new_im = cv2.resize(x1, (180, 180))
            name = 'pic/' +str(bitch)+'_' +str(i) + '.png'
            cv2.imwrite(name, x1)
        b = 0
        name = 'pic/' + str(b) + '_' + str(0) + '.png'
        im = Image.open(name)
        images = []
        for i in range(29):
            name = 'pic/' + str(b) + '_' + str(i + 1) + '.png'
            images.append(Image.open(name))
        im.save('pic/gif.gif', save_all=True, append_images=images, loop=1, duration=1, comment=b"aaabb")


def main(argv=None):  # pylint: disable=unused-argument
    train()


if __name__ == '__main__':
    tf.app.run()






