import numpy as np
def gradient_descent(w_init,lr,steps):
    w=w_init
    losses=[]
    for i in range(steps):
        loss=(w-5)**2
        losses.append(loss)
        grad=2*(w-5)
        w=w-lr*grad
    return losses
for lr in [0.01,0.1,0.5,1.05,0.6]:
    losses=gradient_descent(w_init=0,lr=lr,steps=20)
    print(f"lr={lr}: final loss={losses[-1]:.4f},loss trend={[round(1,0)for l in losses]}")