import numpy as np
x=np.array([0.5,-0.2,0.8])
w=np.array([0.8,1.2,-0.2])
b=-1.2
z=np.dot(x,w)+b
def relu(z):
    return np.maximum(0,z)

output=relu(z)
def leaky_relu(z,alpha=0.01):
    return np.where(z>0,z,alpha*z)
    
print(f"Raw Score(z): {z}")
print(f"Neuron Output: {output}")
print(leaky_relu(z))