import numpy as np
x,y=2,3
w1,w2=0.5,1
b1,b2=0,0

#step 1 Forward Pass
def forwrd_pass(w1,w2,x,b1,b2):
    z1=w1*x+b1
    a1=z1
    y_hat=w2*a1+b2
    return y_hat
y_hat=forwrd_pass(w1,w2,x,b1,b2)
print(f"Prediction: {y_hat}")    
def compute_loss(y_hat,y):
    loss=(y-y_hat)**2
    return loss
loss=compute_loss(y_hat,y)
print(f'Loss: {loss}')
dL_dpred=-2*(y-y_hat)
dy_w2=1     #y_pred=w2*a1+b
dl_w2=dL_dpred*dy_w2
dy_a1=w2
da1_dz1=1
#z1=W1x+b1
dz1_dw1=x
dl_w1=dL_dpred*dy_a1*da1_dz1*dz1_dw1    
print(f"Dl_wrt_W2: {dl_w2}")
print(f"Dl_wrt_W1: {dl_w1}")
n=0.1
w1_new=w1-n*dl_w1
w2_new=w2-n*dl_w2
print(f"W1 new: {w1_new}")
print(f"W2 new: {w2_new}")
