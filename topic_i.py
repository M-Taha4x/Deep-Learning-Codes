import numpy as np
x=np.array([3,2])
w=np.array([3,1])
b=2.4
def neuron(x,w,b,activation='relu'):
    z=np.dot(w,x)+b
    if activation=='relu':
        a=np.maximum(0,z)
    elif activation == 'sigmoid':
        a=1/(1+np.exp(-1))
    elif activation == 'tanh':
        a=np.tanh(z)
    return a
print(neuron(x,w,b,activation='tanh'))