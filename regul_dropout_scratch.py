import numpy as np

def l2_penalty(weights,lam=0.01):
    return lam*np.sum(weights**2)
def dropout(layer_output,drop_rate=0.5,training=True):
    if not training:
        return layer_output
    
    mask=(np.random.rand(*layer_output.shape)>drop_rate).astype(float)
    return layer_output*mask/(1-drop_rate)

weights=np.array([0.5,-1.2,3.0,0.1])
print("L2 penalty: ",l2_penalty(weights))
layer_out=np.array([1.0,2.0,3.0,4.0,5.0])
#print("With dropout (training):",dropout(layer_out,drop_rate=0.5,training=True))

#print("With dropout(Test) :",dropout(layer_out,drop_rate=0.5,training=False))

print('With dropout(Training 0.8): ',dropout(layer_out,drop_rate=0.8,training=True))