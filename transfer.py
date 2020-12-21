import time


import numpy as np
import tensorflow as tf
from PIL import Image
from skimage import color
from skimage.io import imsave


from models.vgg import vgg_16

LEARNING_RATE= 10.0

LOGDIR= './logs/'

FLAGS= tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('content','','Content image')
tf.app.flags.DEFINE_string('style','','Style image.')
tf.app.flags.DEFINE_string('ckpt_file','checkpoints/vgg_16.ckpt','Checkpoint file.')
tf.app.flags.DEFINE_string('result_file','result.jpg','Result file.')
tf.app.flags.DEFINE_string('steps',1000,'Number of steps to run.')
tf.app.flags.DEFINE_string('resize', -1, 'Resize shorter dim of content img to this size.')

IMAGENET_MEAN = [123.68, 116.779, 103.939]

STYLE_LAYERS = ['vgg_16/conv1/conv1_1', 'vgg_16/conv2/conv2_1', 'vgg_16/conv3/conv3_1', 'vgg_16/conv4/conv4_1',
                'vgg_16/conv5/conv5_1']
CONTENT_LAYERS = ['vgg_16/conv4/conv4_2', 'vgg_16/conv5/conv5_2']

CONTENT_WEIGHT = 1
STYLE_WEIGHT = 1e3
TVD_WEIGHT = 1e-2


def resize(img):
    if FLAGS.resize <= 0:
        return img
    width, height = img.size
    ratio = min(width, height) / FLAGS.resize
    if width < height:
        width = FLAGS.resize
        height = int(height / ratio)
    else:
        height = FLAGS.resize
        width = int(width / ratio)
    return img.resize((width, height))



