def conv_output_size(W,F,P,S):
    return (W-F+2*P)//S +1
def pool_output_size(W,F,S):
    return (W-F)//S +1

h,w,c=32,32,3
print(f"Input: {h}, {w}, {c}")
h=conv_output_size(h,F=3,P=1,S=1)
print(f"After Conv1: ({h}, {w}, {c})"); w = h; c = 32
h=pool_output_size(h,F=2,S=2)
print(f"After Pool1: ({h},{w},{c})"); w = h
h=conv_output_size(h,F=3,P=1,S=1)
print(f"After Conv2 ({h}, {w}, {c})"); w = h; c = 64
h=pool_output_size(h,F=2,S=2); w = h
print(f"After Pool2: ({h}, {w}, {c})")
flattened=h*w*c
print(f"After Flatten: ({flattened},)")