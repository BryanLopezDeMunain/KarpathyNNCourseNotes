import math

# We build the Value class from scratch.

class Value:
    def __init__(self, data, _children=(), _op = ''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None # Empty function
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f'Value(data={self.data})'
    
    # Quick investigation into chain rule application to '+' and '*' nodes reveals straightforward backprop logic.
    # For '+' local derivative is 1.0 * thing before.
    # For '*' local derivative w.r.t one thing is data value of all other members of mul * thing before.
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        # If an addition takes place, must define appropriate local backprop
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward # The _backward data member is no longer empty fn. Now contains backprop logic for addition.

        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (float, int))
        out = Value(self.data**other, (self,))

        def _backward():
            self.grad += other * self.data**(other - 1) * out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self, other): # acts as fallback, if a*b fails checks b*a
        return self * other
    
    def __truediv__(self, other):
        out = self * other**(-1)
        return out
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __radd__(self, other): # other + self
        return self + other
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        
        out._backward = _backward

        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,))

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v): # v root node
            if v not in visited:
                visited.add(v) # .add() method defined on sets
                for child in v._prev: # prev is the set of _children of v
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo): # py reversed() calls iterator in reverse order
            node._backward()

    

# Now take it for a spin!

# Acting as nn inputs
x1 = Value(2.0)
x2 = Value(0.0)
# Acting as nn weights
w1 = Value(-3.0)
w2 = Value(1.0)
# Acting as bias for nn
b = Value(6.8813735870195432)

# Now do math
x1w1 = x1 * w1

x2w2 = x2 * w2

x1w1x2w2 = x1w1 + x2w2

n = x1w1x2w2 + b

o = n.tanh()
o.label = 'o'

# grad initialised as zero, chain rule will guarantee 0 grad always
# to solve this we initialise o.grad

o.grad = 1.0
o._backward() # get grad for o
n._backward()
x1w1x2w2._backward()
x1w1._backward()
x2w2._backward()

print(o.grad)
print(n.grad)
print(x1w1x2w2.grad)
print(x1w1.grad)
print(x2w2.grad)
print(x1.grad)
print(w1.grad)
print(x2.grad)
print(w2.grad) # Outputs match Karpathy, no fuck ups

# We want to call backward automatically, need to think about how to enforce the computational graph order.
# Immediately notice sometimes we'll have a choice (e.g. x1w1 first vs x2w2 first).
# We use topological sort.

topo = []
visited = set()

def build_topo(v): # v root node
    if v not in visited:
        visited.add(v) # .add() method defined on sets
        for child in v._prev: # prev is the set of _children of v
            build_topo(child)
        topo.append(v)
build_topo(o)

# Annotated Output:

# Bias: Value(data=6.881373587019543), 
# w1: Value(data=-3.0), 
# x1: Value(data=2.0), 
# w1*x1: Value(data=-6.0), 
# x2: Value(data=0.0), 
# w2: Value(data=1.0), 
# x2*w2: Value(data=0.0), 
# x1*w1 + x2*w2: Value(data=-6.0), 
# x1*w1 + x2*w2 + bias: Value(data=0.8813735870195432), 
# tanh output: Value(data=0.7071067811865476)

# Clearly topological order sensible. Now implement _backward calling in reverse topological order.

# First initialise root node grad.
o.grad = 1.0

for node in reversed(topo): # py reversed() calls iterator in reverse order
    node._backward() # brackets so fn called and logic executed

# Now we just add this code to class.

# Now we break down tanh to illustrate choice of segmentation
# tanh definition contains some + 1s so need to add operation to class def

a = Value(1)
print(a + 1) # Now this works

print(a * 2) # this works because a.__mul__(2) defined
# not the case for 2.__mul__(a) so we define __rmul__

print(2*a) # now good

# We need exp, and division (added through exponentiation)

b = Value(2)
print(a/b)

# Implement subtraction through negation

print(a - b)

# PyTorch implementation straightforward and agrees

# Now neural network implementation

import random

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
class MLP:
    def __init__(self, nin, nouts): # nouts is a list of nout values
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

# Which works
x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])
print('n(x):', n(x))

# make tiny dataset

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]

ys = [1.0, -1.0, -1.0, 1.0]

ypred = [n(x) for x in xs]
print('Pred:', ypred) # terrible outputs

# Implement MSE
loss = sum([(yout - ytrue)**2 for ytrue, yout in zip(ys, ypred)]) # issue here for ytrue - yout because scalar.__sub__(Value) not defined
print('Loss:', loss)

loss.backward()

# n has a list of layers which are each a list of neurons which are each a list of weight (and a bias) and each weight has a grad
print('grad for first weight of first neuron layer 1:', n.layers[0].neurons[0].w[0].grad)

for k in range(20):

    # get predictions, calculate loss
    ypred = [n(x) for x in xs]
    loss = sum((yout - ytrue)**2 for ytrue, yout in zip(ys, ypred))

    # get gradients
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()

    # update weights
    for p in n.parameters():
        p.data += -0.5 * p.grad
    
    print(k, loss.data)

print('ypred:', ypred)