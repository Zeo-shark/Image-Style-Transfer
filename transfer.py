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





