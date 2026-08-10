import numpy as np
def max_pool2D(feature_map,pool_size=2,stride=2,mode='max'):
    h,w=feature_map.shape
    out_h=(h-pool_size)//stride+1
    out_w=(w-pool_size)//stride+1
    
    output=np.zeros((out_h,out_w))
    for i in range(out_h):
        for j in range(out_w):
            row_start=i*stride
            col_start=j*stride
            window=feature_map[row_start:row_start+pool_size,col_start:col_start+pool_size]
            if mode=='max':
                output[i,j]=np.max(window)
            elif mode=='average':
                output[i,j]=np.mean(window)
    return output
fmap=np.array([
    [7,2,5,1],
    [3,9,4,6],
    [1,0,8,2],
    [5,3,1,4]

])
print(max_pool2D(fmap))