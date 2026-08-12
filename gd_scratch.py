import numpy as np
def gradient(w):
    return 2*(w-5)

def gradient_descent(w_init,lr=0.1,steps=20):
    w=w_init
    history=[w]
    for i in range(steps):
        print(f"{history[i]:.2f}")
        grad=gradient(w)
        w=w-lr*grad
        history.append(w)
    return history

history=gradient_descent(w_init=0,lr=0.1,steps=20)
print(f"Final w: {history[-1]:.4f}")
print(f"First 5 steps: {[round(h,3)for h in history[:5]]}")