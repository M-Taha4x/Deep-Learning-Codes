import numpy as np
def conv2d(image,kernel,stride=2,padding=0):
    if padding>0:
        image=np.pad(image,padding,mode='constant')
    img_h,img_w=image.shape
    k_h,k_w=kernel.shape
    
    out_h=(img_h-k_h)//stride+1
    out_w=(img_w-k_w)//stride+1
    output=np.zeros((out_h,out_w))
    for i in range(0,out_h):
        for j in range(0,out_w):
            row_start=i*stride
            col_start=j*stride
            patch=image[row_start:row_start+k_h,col_start:col_start+k_w]
            output[i,j]=np.sum(patch*kernel)
            
    return output
image=np.array([
    [1,1,1,0,0],
    [0,1,1,1,0],
    [0,0,1,1,1],
    [0,0,1,1,0],
    [0,1,1,0,1]
])
kernel=np.array([
    [1,0,1],
    [0,1,0],
    [1,0,1]  
])
print(conv2d(image,kernel))