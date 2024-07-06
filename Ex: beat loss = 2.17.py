# Karpathy leaves as an exercise to beat loss = 2.17 using the video 3 version of makemore.

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))

g = torch.Generator().manual_seed(2147483647)

block_size = 3
n_dim = 30

def build_dataset(words):
    X, Y = [], []

    stoi = {s:i + 1 for i, s in enumerate(chars)}
    stoi['.'] = 0

    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

import random
random.seed(42)
random.shuffle(words)

n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

X_train, Y_train = build_dataset(words[:n1])
X_dev, Y_dev = build_dataset(words[n1:n2])
X_test, Y_test = build_dataset(words[n2:])

C = torch.randn((27, n_dim), generator=g)
W1 = torch.randn((n_dim*block_size, 200), generator=g)
b1 = torch.randn(200, generator=g)
W2 = torch.randn((200, 27), generator=g)
b2 = torch.randn(27, generator=g)

parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad = True

n_iters = 100000
batch_size = (256,) # must be tuple

lre =  torch.linspace(-3, 0, n_iters)
lrs = 10**lre

lri, lossi, stepi = [], [], []

for i in range(n_iters):
    ix = torch.randint(0, X_train.shape[0], batch_size)

    emb = C[X_train[ix]]
    h = torch.tanh(emb.view(-1, n_dim*block_size) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y_train[ix])

    for p in parameters:
        p.grad = None

    loss.backward()

    lr = 0.1 if 1 < 50000 else 0.01

    for p in parameters:
        #p.data += -lrs[i] * p.grad
        p.data += -lr * p.grad

    lri.append(lre[i])
    lossi.append(loss.log10().item())
    stepi.append(i)

emb = C[X_dev]
h = torch.tanh(emb.view(-1, n_dim*block_size) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Y_dev)
print(loss.item())

# batch size 400 gives loss = 2.17 but takes a while to run.
# batch size 256 but 50k iterations after lr decay gives loss = 2.16.

