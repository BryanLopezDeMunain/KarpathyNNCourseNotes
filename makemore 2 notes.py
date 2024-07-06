import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words =open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))

stoi = {s: i + 1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

# to build the dataset

block_size = 3 # context size
X, Y = [], []

for w in words[:5]:
    #print(w)
    context = [0] * block_size # initialise context window as bunch of zeroes
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        print(''.join(itos[i] for i in context), '->', itos[ix])
        context = context[1:] + [ix]

X = torch.tensor(X)
Y = torch.tensor(Y)

g = torch.Generator().manual_seed(2147483647)

C = torch.randn((27, 2), generator=g)

print(C[5]) # prints embedding of 5th char

print(F.one_hot(torch.tensor(5), num_classes=27).float() @ C)
# prints same result

# We can index with a list or tensor
print(C[X]) # returns embeddings of X in 3D tensor

# we embed all X with
emb = C[X]

W1 = torch.randn((6, 100), generator=g) # 6 because 3 2D embeddings
b1 = torch.randn(100, generator=g)
print(b1.shape)

# we concatenate emb to do emb @ W1 + b1
print(emb.view(-1, 2*block_size)[0])
h = torch.tanh(emb.view(-1, 2*block_size) @ W1 + b1)

W2 = torch.randn((100, 27), generator=g)
b2 = torch.randn(27, generator=g)

parameters = [C, W1, b1, W2, b2]

logits = h @ W2 + b2

counts = logits.exp()
prob = counts / counts.sum(1, keepdim=True)
loss = -prob[torch.arange(32), Y].log().mean()

loss = F.cross_entropy(logits, Y) # does same thing

print(loss) # Loss: 17.77

# Let's train and on the full dataset
X, Y = [], []

for w in words:
    #print(w)
    context = [0] * block_size # initialise context window as bunch of zeroes
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        context = context[1:] + [ix]

X = torch.tensor(X)
Y = torch.tensor(Y)
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 2), generator=g)

W1 = torch.randn((6, 100), generator=g) # 6 because 3 2D embeddings
b1 = torch.randn(100, generator=g)
W2 = torch.randn((100, 27), generator=g)
b2 = torch.randn(27, generator=g)

parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad = True

lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre

lri, lossi, stepi = [], [], []

for i in range(1000):
    # get minibatch indices
    ix = torch.randint(0, X.shape[0], (32,))

    # forward pass
    emb = C[X[ix]]
    h = torch.tanh(emb.view(-1, 2*block_size) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y[ix])

    for p in parameters:
        p.grad = None
    
    loss.backward()
    
    for p in parameters:
        p.data += -lrs[i] * p.grad

    lri.append(lre[i])
    stepi.append(i)
    lossi.append(loss.log10().item())

print(loss.item())

# orders of magnitude faster with minibatching
# better loss with decaying learning rate