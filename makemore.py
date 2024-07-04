words = open('names.txt', 'r').read().splitlines() # list of strings (words)

print(len(words)) # 32033 words

print('Shortest',min(len(w) for w in words)) # shortest word length 2

print(max(len(w) for w in words)) # longest word length 15

# Start by building a bigram language model

for w in words[:3]:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        print(ch1, ch2)

# We want to learn stats about which characters are likely to follow other chars
# we do this by counting.
b = {} # empty dict
for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1

sorted(b.items(), key = lambda kv: -kv[1]) # kv is a tuple say (('q', 'r'), 1), we want to sort by value of second element in big tuple

# We want this info in a 2d array not a dict
import torch

# We want to work with torch.int32 because we're counting
N = torch.zeros((28, 28), dtype=torch.int32)

chars = sorted(set(''.join(words)))

stoi = {s:i for i,s in enumerate(chars)}
stoi['<S>'] = 26
stoi['<E>'] = 27

for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1 #  can be interpreted as a sum of adjacency matrices for each word

# worth noting last row is completely zeroes because N[28, some character] refers to cases where ('<E>', some character). In individual words this never happens because <E> is at the end.

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

itos = {i:s for s,i in stoi.items()}

#plt.figure(figsize=(16,16))
#plt.imshow(N, cmap='Blues')
#for i in range(28):
#    for j in range(28):
#        chstr = itos[i] + itos[j]
#        plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
#        plt.text(j, i, N[i,j].item(), ha='center', va='top', color='gray')
#plt.axis('off')
#plt.show()

# We want to clean this up, we start by getting rid of <S> and <E> and replacing with .

N = torch.zeros((27, 27), dtype=torch.int32)
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

# Now we can define probabilities
# N[0] is the top row, contains info about how many times a word starts with each character

p = N[0].float()
p = N[0] / p.sum()

g = torch.Generator().manual_seed(2147483647)
#for i in range(10):
#    out = []
#    ix = 0
# 
#    while True:
#        p = N[ix]
#        p = p.float()
#        p = p / p.sum()
#        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
#        out.append(itos[ix])
#        if ix == 0:
#            break
#    print(''.join(out))

# Names are terrible
# Single letter names occur when randomly generated letter is terrible (read common ending letter)

# we prepare a P that stores probs to avoid recalculation
P = torch.zeros((27, 27), dtype=torch.float32)

# my naive attempt
for ix in range(27):
    p = N[ix].float()
    P[ix] += p / p.sum()

# Karpathy's vectorised operations
P = N.float()
P /= P.sum(1, keepdims=True) # use in-place \= to consider memory (instead of creating new tensor and reassigning)

for i in range(5):
    out = []
    ix = 0
 
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))

# How do we evaluate the quality of the model?
# we define the negative log likelihood and implement model smoothing to avoid inf results

P = (N + 1).float()
P /= P.sum(1, keepdims=True)

log_likelihood = 0.0
n = 0

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]

        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1

nll = -log_likelihood
print(f'{log_likelihood=}')
print(f'{nll=}')
print(f'{nll/n}')

# nll/n is a good measure of quality. We seek to minimise this now
# we recast the problem into a form compatible with a nn
# we start by building a dataset of bigrams

xs, ys = [], []

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]

        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)

# now how do we feed these into a nn?
# we one hot encode with torch.nn.functional.one_hot(tensor, num_classes)

import torch.nn.functional as F

xenc = F.one_hot(xs, num_classes=27)
# print(xenc.shape) returns torch.Size([228146, 27])

# need to make onehot entries float32
xenc = F.one_hot(xs, num_classes=27).float()

# we now construct a neuron
W = torch.randn((27, 1))

xenc @ W # is the (5,1) output for a single neuron

# we want 27 sets of weights (27 neurons) so we make W 27 x 27

W = torch.randn((27, 27)) # this is now a layer

xenc @ W # is the (5,27) output of a full layer

# every element (a, b) in xenc @ W is the firing rate of the bth neuron
# on the ath input for a in {1,2,3,4,5} in his example

# we keep the model 1 layer for now
# we want the outputs to be interpretable as character predictions
# we interpet xenc @ W to be log-counts (logits), so we exponentiate

logits = xenc @ W
counts = logits.exp()
prob = counts / counts.sum(1, keepdim=True)
# print(prob.shape) returns torch.Size([228146, 27])

# every probs[i] is the neural net's assignment for how likely each of the 27 characters are to come after i

# Summarise the scaffolding we built
W = torch.randn((27,27), generator=g, requires_grad=True) # for reproducibility
xenc = F.one_hot(xs, num_classes=27).float()
logits = xenc @ W
counts = logits.exp()
probs = counts / counts.sum(1, keepdims=True)

# implement nn
num = xs.nelement()
for k in range(100):

    # do forward pass
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True)
    loss = -probs[torch.arange(num), ys].log().mean() + 0.1*(W**2).mean() # L1 regularisation (encourages sparse representation)

    # do backward pass
    W.grad = None
    loss.backward()

    # update weights
    W.data += -50 * W.grad

    if k % 10 == 0:
        print(loss.item())

# final loss is 2.4754 without L1
#               2.59 with L1 for reg strength 0.1

# To sample from nn
for i in range(5):
    out = []
    ix = 0
    while True:
        # forward pass
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdim=True)

        # get character from prob dist
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))

# Output:
# aliaiadayraufbrlqdouseyton.
# zqzevaran.
# han.
# ke.
# etssleleronakfn.

# Terrible but much better than random.
# Need larger database, more layers.
