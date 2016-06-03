import tensorflow as tf
import numpy as np
from collections import OrderedDict
import negsamp
import sys
import matplotlib.pylab as plt

# Params
d = 100
f = 4096
V = 291
m = 10
lr = 0.001
epochs = 100
weightfile = None

# Load the data
if True:
  data = np.load('data/iaprtc_alexfc7.npz')
  D = open('data/iaprtc_dictionary.txt').read().splitlines()
  train_ims = [ im.split('/')[-1] for im in open('data/iaprtc_trainlist.txt').read().splitlines() ]
  test_ims = [ im.split('/')[-1] for im in open('data/iaprtc_testlist.txt').read().splitlines() ]
  xTr = data['xTr'].T
  yTr = data['yTr'].T
  xTe = data['xTe'].T
  yTe = data['yTe'].T
  wc = yTr.sum(axis=0)+0.01-0.01

with tf.InteractiveSession() as sess:

  # Placeholders for inputs and outputs
  x = tf.placeholder(tf.float32, shape=[None, 4096])
  y_ = tf.placeholder(tf.float32, shape=[None, 291])

  if weightfile:
    print "Not implemented"
  else:
    Wi = tf.truncated_normal( [d, f] )
    Wc = tf.truncated_normal( [V, d] )

  cost_relax = tf.reduce_sum( tf.pow( tf.sub(sr_relax[i], y), 2 ) )

