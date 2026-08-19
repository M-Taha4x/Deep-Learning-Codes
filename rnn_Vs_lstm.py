import numpy as np
def sigmoid(z):
    return 1/(1+np.exp(-z))
def rnn_cell(h_prev,x,W_h,W_x,b):
    #h_prev=Past mem,x=curr input,
    #W_h=hidden Weights
    #W_x=input weights
    #b=bias
    return np.tanh(W_h*h_prev+W_x*x+b)
def lstm_cell(h_prev,C_prev,x,weights):
    Wf,Wi,Wc,Wo,bf,bi,bc,bo=weights
    combined=h_prev+x
    f_t=sigmoid(Wf*combined+bf) #Forget Gate
    print(f"F_T: {f_t:.4f}")
    i_t=sigmoid(Wi*combined+bi) #Input gate
    C_tilde=np.tanh(Wc*combined+bc) #Candidate Mem
    C_t=f_t*C_prev+i_t*C_tilde #Cell State Update
    o_t=sigmoid(Wo*combined+bo) #Output Gate
    h_t=o_t*np.tanh(C_t)
    return h_t,C_t

h_prev,C_prev,x=0.5,0.3,1.0
weights=(0,0.6,0.4,0.3,0,0,0,0)#Wf,Wi,Wc,Wo,bf,bi,bc,bo
h_new,C_new=lstm_cell(h_prev,C_prev,x,weights)
print(f"New Hidden State: {h_new:.4f}, New cell State: {C_new:.4f}")