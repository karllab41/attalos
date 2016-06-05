import theano.tensor as T
import theano
import numpy as np
from collections import OrderedDict
import negsamp
import sys
import matplotlib.pylab as plt

# Params
d = 100
m = 10
lr = 0.001
epochs = 100
weightfile = 'params-i.npz'

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
  d = 100
  f = 4096
  V = 291
  test_ims_full = [ im for im in open('data/iaprtc_testlist.txt').read().splitlines() ]
if weightfile:
  Wi = theano.shared(np.load(weightfile)['Wi'])
  Wc = theano.shared(np.load(weightfile)['Wc'])

# Define functionality
x = T.vector()
p = T.ivector()
n = T.ivector()
yp = Wc[p].dot(Wi.dot(x))
yn = Wc[n].dot(Wi.dot(x))
loss = -T.log(T.nnet.sigmoid(yp)).mean() - T.log(T.nnet.sigmoid(-yn)).mean()

# Define testing procedure
yh = T.nnet.sigmoid( Wc.dot(Wi.dot(x)) )
predictor = theano.function( inputs =[x], outputs = yh, allow_input_downcast=True )

# Run example
i=1339;
input=xTe[i];output=yTe[i];imname='images/'+test_ims_full[i]+'.jpg';im=plt.imread(imname)
ypwords = [D[j] for j in predictor( input ).argsort()[::-1] [ 0:(predictor(input)>0.5).sum() ] ]
ytwords = [D[j] for j in ns.posidx( output ) ]

