import torch
import math
import colorsys

# WaterCan helpers
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

def get_from(dict,key_prop):
    return {key:perform[key_prop] for key,perform in dict.items()} 

# Visualization
def generate_tuples(n):
    side_length = math.ceil(math.sqrt(n))
    tuples_list = [(i // side_length, i % side_length) for i in range(n)]
    return tuples_list


def generate_colors(n):
    colors = []
    for i in range(n):
        hue = i / n  # Evenly space colors by hue
        saturation = 0.5 + 0.5 * (i % 2)  # Alternate saturation for more distinct colors
        value = 0.95  # Keep the value high for brightness
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(tuple(rgb))
    return colors


## Method
def min_with_index(lst):
    # Find the minimum value and its index
    min_val, min_arg = min((val, idx) for idx, val in enumerate(lst))
    return min_val, min_arg