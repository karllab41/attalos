# Evaluation Class

## How to run the MATLAB code from INRIA/LEAR
Utility code was downloaded from [iMin-Li's website](http://www1.cse.wustl.edu/~mchen/code/FastTag/fasttag.tar.gz) but she likely got it from an evaluation website. You will need oct2py (pip or conda install it). 

Make sure you have input as 2D numpy arrays. For example, if I have numpy arrays `y_test` and `y_truth` as two arrays I want to compare.

```
from oct2py import octave
octave.addpath('/this/directory/')
[precision, recall, f1score] = octave.evaluate(y_truth, t_test, 5)

```

## How to run the Python Class

