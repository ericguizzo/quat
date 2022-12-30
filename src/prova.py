import torch 
import numpy as np

device = "cpu"
def dyn_pad(input, device, x_source, x_target):
    #dynamic move in time desired portion of sound 
    #this this is because zeropadding is always added to the end
    input = input[:,:,:x_source,:]
    b,c,x,y = input.shape
    pad = torch.zeros(b,c,x_target,y).to(device)
    diff = x_target - x - 1
    random_init = np.random.randint(diff)
    pad[:,:,random_init:random_init+x,:] = input

    return pad


a = torch.zeros(1,1,10,10)
a[:,:,:2,:] = 1

b = dyn_pad(a, device, 2, 10)
print (b)