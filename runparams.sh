#!/bin/bash

prefix='2048,1024,200'
batchsize=256
bugsamp=''
# bugsamp='--bugsamp'

# PYTHONPATH=$PWD/ python2 attalos/imgtxt_algorithms/regress2sum/regress2sum.py    /local_data/teams/attalos/features/image/iaprtc_train_20160816_inception.hdf5    /local_data/teams/attalos/features/text/iaprtc_train_20160816_text.json.gz    /local_data/teams/attalos/features/image/iaprtc_test_20160816_inception.hdf5      /local_data/teams/attalos/features/text/iaprtc_test_20160816_text.json.gz /local_data/yonas/glove.6B.200d.txt    --network 2048,1024 --learning_rate 1e-4 --batch_size=128 --epochs=250 --in_memory --model_type=negsampling --optim_words &> joint-1024.txt      

outfile=$prefix-${batchsize}$bugsamp-optim.txt
PYTHONPATH=$PWD/ python2 attalos/imgtxt_algorithms/regress2sum/regress2sum.py \
  /local_data/teams/attalos/features/image/iaprtc_train_20160816_inception.hdf5    \
  /local_data/teams/attalos/features/text/iaprtc_train_20160816_text.json.gz    \
  /local_data/teams/attalos/features/image/iaprtc_test_20160816_inception.hdf5    \
  /local_data/teams/attalos/features/text/iaprtc_test_20160816_text.json.gz    \
  /local_data/yonas/glove.6B.200d.txt \
  --network $prefix --learning_rate 1e-4 --batch_size=$batchsize --optim_words \
  --epochs=250 --model_type=negsampling $bugsamp &> $outfile &
#  --word_vector_type=glove --epoch_verbosity=1 --verbose_eval --optim_words &> $outfile & 

outfile=$prefix-${batchsize}$bugsamp-fixed.txt
PYTHONPATH=$PWD/ python2 attalos/imgtxt_algorithms/regress2sum/regress2sum.py \
  /local_data/teams/attalos/features/image/iaprtc_train_20160816_inception.hdf5 \
  /local_data/teams/attalos/features/text/iaprtc_train_20160816_text.json.gz \
  /local_data/teams/attalos/features/image/iaprtc_test_20160816_inception.hdf5 \
  /local_data/teams/attalos/features/text/iaprtc_test_20160816_text.json.gz \
  /local_data/yonas/glove.6B.200d.txt \
  --network $prefix --learning_rate 1e-4 --batch_size=$batchsize \
  --epochs=250 --model_type=negsampling $bugsamp &> $outfile & 
#  --word_vector_type=glove --epoch_verbosity=1 --verbose_eval --optim_words &> $outfile &                                          

