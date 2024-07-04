# This code is written by me, but heavily based on Andrej Karpathy's second
# video in his YouTube course Neural Networks: Zero to Hero.

#E01: train a trigram language model, i.e. take two characters as an input 
# to predict the 3rd one. 
# Feel free to use either counting or a neural net. 
# Evaluate the loss; Did it improve over a bigram model?

import torch

# Import data (32.033k names)
words = open('names.txt', 'r').read().splitlines()

#---------------------------------------------------------------------------#
# Counting approach
#---------------------------------------------------------------------------#

# Separate into characters
N = torch.zeros((27, 27, 27), dtype=torch.int32)

chars = sorted(set(''.join(words))) # returns the alphabet
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        ix3 = stoi[ch3]

        N[ix1, ix2, ix3] += 1

g = torch.Generator().manual_seed(2147483647)

P = torch.zeros((27, 27, 27))

P = (N + 1).float() # +1 for slight model smoothing
P /= P.sum(1, keepdims=True) # changing keepdims can mess up broadcasting without throwing error

log_likelihood = 0.0
n = 0

# Evaluate trigram model on entire training dataset
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        ix3 = stoi[ch3]

        prob = P[ix1, ix2, ix3]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1

nll = -log_likelihood
nll_norm = nll / n
print(f'Normalised negative log likelihood: {nll_norm.item()}')

# Loss is better than bigram model

