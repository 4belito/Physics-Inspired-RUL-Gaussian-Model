import torch


def derivate(f,x):
    return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True)[0] 

def high_derivate(f,x,ord,cumul):
    if cumul: derivatives=[f]
    for _ in range(ord):
        f=derivate(f,x)
        if cumul: derivatives.append(f)
    if cumul:
        return derivatives
    else:
        return f

def derivate_vector(fs,x):
    fs_d=[]
    for f in fs:
        fs_d.append(derivate(f,x))
    return fs_d

def gpu2np(x):
    return x.cpu().detach().numpy().squeeze()