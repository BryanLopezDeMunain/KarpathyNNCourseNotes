#E01: train a trigram language model, i.e. take two characters as an input 
# to predict the 3rd one. 
# Feel free to use either counting or a neural net. 
# Evaluate the loss; Did it improve over a bigram model?

import torch

# Import data (32.033k names)
words = open('names.txt', 'r').read().splitlines()

# Separate into characters
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):

