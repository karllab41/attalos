import theano.tensor as T
import theano
import numpy as np
from collections import OrderedDict
import attalos.imgtxt_algorithms.util.negsamp as negsamp
import sys
import matplotlib.pylab as plt

# Params
d = 100
m = 20
lr = 0.01
epochs = 100
batchsize=256
weightfile = None # 'params-2layer.npz'
pretrain = False

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
  hidden = 4096
  V = 291
  if weightfile and pretrain:
    Wi = theano.shared(np.load(weightfile)['Wi'])
    Wh = theano.shared(np.random.ranf((hidden, f)))
    Wc = theano.shared(np.load(weightfile)['Wc'])
  elif weightfile:
    Wi = theano.shared(np.load(weightfile)['Wi'])
    Wh = theano.shared(np.load(weightfile)['Wh'])
    Wc = theano.shared(np.load(weightfile)['Wc'])
  else:
    Wh = theano.shared(np.random.ranf((hidden, f)))
    Wi = theano.shared(np.random.ranf((d, hidden)))
    Wc = theano.shared(np.random.ranf((V, d)))

# Negative sampler
ns = negsamp.NegativeSampler(wc / wc.sum())

# Define functionality
x = T.matrix()
p = T.matrix()
n = T.matrix()

# Cross correlation
xcorr = Wc.dot(Wi.dot(T.nnet.relu(Wh.dot(x.T)))) 

# loss = -T.log(T.nnet.sigmoid(p.dot(xcorr))).mean() - T.log(T.nnet.sigmoid(-n.dot(xcorr))).mean()
loss = -T.log(T.nnet.sigmoid( (p-n).dot(xcorr)  )).mean()

# Define the gradient updates. Use positive for maximization
params = [Wi, Wc, Wh]
gWi, gWc, gWh = T.grad(loss, params)
sgd = OrderedDict( { Wi: Wi - lr*gWi, Wc: Wc - lr*gWc, Wh: Wh - lr*gWh } )

# Compile to theano functionality
train = theano.function( [x,p,n], outputs=loss, updates=sgd, allow_input_downcast=True )

# Iterate through the data size
for j in xrange(epochs):
  print "Epoch "+str(j)
  k=0
  totloss = 0.0
  batloss = 0.0
  randorder = np.random.permutation(len(yTr))
  for i in range(0,len(randorder),batchsize):
    indata = xTr[i:i+batchsize]
    outdata= yTr[i:i+batchsize]

    lossval = train( indata, outdata, ns.negsampv(outdata, m) )
    totloss += lossval
    batloss += lossval
    k+=1
    if k % 4 == 0:
      sys.stdout.write("Iter:"+str(k*batchsize)+", Batloss="+str(batloss)+" Samploss="+str(batloss/batchsize)+"\r")
      sys.stdout.flush()
      batloss=0.0
  print ""
  print "Total loss on epoch "+str(j)+" = "+str(totloss)
  np.savez('params-2layer_relu.npz', Wi=Wi.get_value(), Wh=Wh.get_value(), Wc=Wc.get_value(), Epoch=j)

# Define testing procedure
xv = T.vector()
yh = T.nnet.sigmoid( Wc.dot(Wi.dot(T.nnet.relu(Wh.dot(xv)))) )
predictor = theano.function( inputs =[xv], outputs = yh, allow_input_downcast=True )

