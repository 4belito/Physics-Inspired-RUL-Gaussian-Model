
import torch
import numpy as np
import os

def derivate(f,x):
    return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True)[0]

def derivate_vector(fs,x):
    fs_d=[]
    for f in fs:
        fs_d.append(derivate(f,x))
    return fs_d

def high_derivate(f,x,ord,cumul):
    if cumul: derivatives=[f]
    for _ in range(ord):
        f=derivate(f,x)
        if cumul: derivatives.append(f)
    if cumul:
        return derivatives
    else:
        return f



def gpu2np(x):
    return x.cpu().detach().numpy().squeeze()

def np2gpu(x,device):
    x=torch.tensor(x, dtype=torch.float32).view(-1, 1).to(device)#.double()
    return x

def get_device(model):
    return next(model.parameters()).device

def extend(x,extra_time):
    len_x=len(x)
    train_step=(x[-1]-x[0])/len_x
    extension=np.arange(x[-1],x[-1]+extra_time,train_step)
    x_ext=np.concatenate([x,extension[1:]])
    return x_ext


def check_weights(file_path):
    if os.path.isfile(file_path):
        response = input(f"The weights '{file_path}' already exists. Do you want to continue the training? (y/n): ")
        if response.lower() == 'y':
            print("loading weights...")
            return True
        else:
            print("overwriting weights")
            return False
    else:
        return False
    

