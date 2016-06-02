import theano.tensor as T
import theano
import numpy as np
from collections import OrderedDict
import attalos.imgtxt_algorithms.util.negsamp as negsamp
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
  if weightfile:
    Wi = theano.shared(np.load(weightfile)['Wi'])
    Wc = theano.shared(np.load(weightfile)['Wc'])
  else:
    Wi = theano.shared(np.random.ranf((d, f)))
    Wc = theano.shared(np.random.ranf((V, d)))

# Negative sampler
ns = negsamp.NegativeSampler(wc / wc.sum())

# Define functionality
x = T.vector()
p = T.ivector()
n = T.ivector()
yp = Wc[p].dot(Wi.dot(x))
yn = Wc[n].dot(Wi.dot(x))
loss = -T.log(T.nnet.sigmoid(yp)).mean() - T.log(T.nnet.sigmoid(-yn)).mean()

# Define the gradient updates. Use positive for maximization
params = [Wi, Wc]
gWi, gWc = T.grad(loss, params)
sgd = OrderedDict( { Wi: Wi - lr*gWi, Wc: Wc - lr*gWc } )

# Compile to theano functionality
train = theano.function( [x,p,n], outputs=loss, updates=sgd, allow_input_downcast=True )

# Iterate through the data size
for j in xrange(epochs):
  print "Epoch "+str(j)
  totloss = 0.0
  batloss = 0.0
  k = 0
  for i in np.random.permutation(len(yTr)):
    indata = xTr[i]
    outdata= yTr[i]
    lossval = train( indata, ns.posidx(outdata), ns.negsamp(outdata, m) )
    totloss += lossval
    batloss += lossval
    k+=1
    if k % 256 == 0:
      sys.stdout.write("Iteration "+str(j)+" Batch loss="+str(batloss)+"\r")
      sys.stdout.flush()
      batloss=0.0
  print ""
  print "Total loss on epoch "+str(j)+" = "+str(totloss)
  np.savez('params-i.npz', Wi=Wi.get_value(), Wc=Wc.get_value(), Epoch=j)

# Define testing procedure
yh = T.nnet.sigmoid( Wc.dot(Wi.dot(x)) )
predictor = theano.function( inputs =[x], outputs = yh, allow_input_downcast=True )

# Run example
i=1339;
input=xTe[i];output=yTe[i];imname='images/'+test_ims[i]+'.jpg';im=plt.imread(imname)
ypwords = [D[j] for j in predictor( input ).argsort()[::-1] [ 0:(predictor(input)>0.5).sum() ] ]
ytwords = [D[j] for j in ns.posidx( output ) ]


