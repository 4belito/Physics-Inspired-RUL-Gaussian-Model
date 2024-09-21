import torch

def get_symmetric_extension(x_full,device,factor_len_ext=0,n_points_ext=1000): 
    max_value=torch.max(x_full).item()
    min_value=torch.min(x_full).item()
    interval_len=(max_value-min_value)
    end_ext=max_value+factor_len_ext*interval_len
    ini_ext=min_value-factor_len_ext*interval_len
    extension=torch.linspace(ini_ext,end_ext,n_points_ext,requires_grad=True,device=device)
    return extension.unsqueeze(-1)

def get_future_extension(x_full,device,len_ext,repeat=1.):
    max_value=torch.max(x_full).item()
    min_value=torch.min(x_full).item()
    interval_len=(max_value-min_value)
    end_ext=max_value+repeat*interval_len
    ini_ext=max_value
    extension=torch.linspace(ini_ext,end_ext,len_ext,requires_grad=True,device=device)
    return extension.unsqueeze(-1)