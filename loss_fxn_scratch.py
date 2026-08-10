import numpy as np
def mse(y_actual,y_pred):
    return np.mean((y_actual-y_pred)**2)
def mae(y_actual,y_pred):
    return np.mean(np.abs(y_actual-y_pred))
def binary_cross_entropy(y_actual,y_pred,epsilon=1e-15):
    y_pred=np.clip(y_pred,epsilon,1-epsilon)
    return -np.mean(y_actual*np.log(y_pred)+(1-y_actual)*np.log(1-y_pred))
def categorical_cross_entropy(y_actual,y_pred,epsilon=1e-15):
    y_pred=np.clip(y_pred,epsilon,1-epsilon)
    return -np.sum(y_actual*np.log(y_pred))

print(mse(np.array([3,5,2]),np.array([2.5,5.5,4])))
print(binary_cross_entropy(np.array([1]),np.array([0.8])))
print(categorical_cross_entropy(np.array([1,0,0]),np.array([0.7,0.2,0.1])))
print(mae(np.array([3,5,2]),np.array([1,3,2])))
print(binary_cross_entropy(np.array([1]), np.array([0.99])))
print(binary_cross_entropy(np.array([1]), np.array([0.01])))