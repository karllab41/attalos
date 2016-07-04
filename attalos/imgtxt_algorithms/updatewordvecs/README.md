# Update word vectors given vector files

This is built off the original Mikolov code at http://code.google.com (though, it appears to be offline as of this moment.)

The main goal of this code is to enable the user to fix a few vectors so that the word space can start to coalesce around these specified vectors.

### Major differences

1. Now, we can upload word vectors (vin and vout, from the original paper).
  - This is done with the option --read-vecs
2. You can upload a dictionary of words (each line is a word/phrase) that you wish to remain static.
  - Note, only vout remains static. We allow vin to change.
  - This is done with the option --read-imvocab
3. You can save vout now. (Previously, you could only save vin.)
  - Note, you specify only the prefix, and not the full filename.

Note: The differences can only be seen when you are doing negative sampling.
