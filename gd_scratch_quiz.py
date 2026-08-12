def gradient_descent(w_init,lr,steps):
    w=w_init
    for i in range(steps):
        loss=(w-5)**2
        print(f"Step: {i} W: {w:.3f} Loss {loss:.3f}")
        grad=2*(w-5)
        w=w-lr*grad
        

print("Learning Rate:0.1 \n")
gradient_descent(0,0.1,5)
print("\n Learning rate:1.1 \n")
gradient_descent(0,1.1,5)
        