import torch
import numpy as np
from scipy.optimize import minimize


def compute_eval_matrices(group_distributions, times):
    A,S = {},{}
    for perform_name in group_distributions.keys():
        A[perform_name]=[]
        S[perform_name]=[]
        for time in times:
            out=group_distributions[perform_name].forward_single(time)
            A[perform_name].append(out[:,:,0].cpu().detach().numpy())
            S[perform_name].append((out[:,:,1]**2).cpu().detach().numpy())
    return A,S



def euclidean_norm_constraint(x):
    return np.sum(x**4)-1

## linear combination methods
def linear_combination_center_mean(x,n): #n=len(x)
    alpha=x[:-1]
    eps=x[-1]
    return alpha+(1+eps-np.sum(alpha))/n 

def linear_combination_center_vector(x,i): #the base depends on the selected believe
    x_reduced = np.delete(x, i)
    x[i]-=np.sum(x_reduced)
    return x


## sample methods
def sample_uniform_conv_comb(n,std): # this works with linear_combination_center_mean
    u = np.sort(np.random.uniform(0, 1, n-1))
    alpha0 = np.concatenate(([u[0]], np.diff(u), [1 - u[-1]]))
    alpha0_noisy = alpha0 + np.random.normal(0, std, n)
    return np.append(alpha0_noisy, 0.)

def sample_strong_believe(n,std): # this works with linear_combination_center_mean
    i = np.random.randint(n)
    alpha0 = np.zeros(n)
    alpha0[i] = 1.
    alpha0_noisy = alpha0 + np.random.normal(0, std, n)
    return np.append(alpha0_noisy, 0.)

def sample_gaussian_KDE(n,std,vectors): # this works with linear_combination_center_vector
    i = np.random.randint(n)
    x0 = np.zeros(n)
    x0[i] = 1.
    norms_i=np.linalg.norm(vectors-vectors[i],axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        std=np.where(norms_i != 0, std / norms_i, 0) 
    x0 += np.random.normal(0, std, n)
    return x0,i

## Objective
def objective(x, A, b, S,gamma,x0):
    return np.sum((A.dot(x) - b)**2/S.dot(x**2)) +np.sum(np.log(S.dot(x**2)))  + gamma* (np.sum((x-x0)**2))

def convex_objective(x, A, b, S,gamma,x0,n):
    alpha=linear_combination_center_mean(x,n)
    #alpha=linear_combination_center_vector(x,n)
    return np.sum((A.dot(alpha) - b)**2/S.dot(alpha**2)) +np.sum(np.log(S.dot(alpha**2)))  + gamma* (np.mean((x-x0)**2))

def mc_simulation(n_paths, n_train_unit, A_n, b_n,Sn, gamma, t, group_network,ood_coef):       
    preds_mean,preds_vars,alphas=[],[],[]
    for _ in range(n_paths):
        #x0 = sample_uniform_conv_comb(n_train_unit,std=ood_coef)
        x0 = sample_strong_believe(n_train_unit,std=ood_coef)
        #x0,i=sample_gaussian_KDE(n_train_unit,std=ood_coef,vectors=A_n.T)
        constr = {'type': 'eq', 'fun': euclidean_norm_constraint}
        x=minimize(convex_objective, x0=x0, args=(A_n, b_n,Sn, gamma,x0,n_train_unit),constraints=constr).x
        # options = {
        #     'maxiter': 1000,
        #     'disp': True,
        #     'gtol': 1e-6,
        #     'xatol': 1e-6
        #     }
        #x=minimize(convex_objective, x0=x0, args=(A_n, b_n,Sn, gamma,x0,i)).x
        
        #alpha=linear_combination_center_vector(x,n_train_unit)
        alpha=linear_combination_center_mean(x,n_train_unit)
        #alpha=linear_combination_center_vector(x,i)
        alpha=torch.tensor(alpha, dtype=torch.float32,device=group_network.device)
        t=torch.tensor(t, dtype=torch.float32,device=group_network.device)
        # alpha=minimize(objective, x0=alpha0, args=(A_n, b_n, Sn, gamma,alpha0),bounds=bounds).x
        pred_mean,pred_vars=group_network.forward_gaussian_lc(t,alpha,out_dist=False,out_std=False)
        preds_mean.append(pred_mean)
        preds_vars.append(pred_vars)
        #alphas.append(alpha)
    preds_mean=torch.stack(preds_mean)
    preds_var=torch.stack(preds_vars)
    #alpha_init=np.stack(alphas)
    return preds_mean, preds_var#, alpha_init


def compute_performance_distribution(t,gamma,group_network,An,bn,Sn,n_train_unit,n_paths=10,ood_coef=0):
    preds_mean, preds_vars = mc_simulation(n_paths, n_train_unit, An, bn,Sn, gamma, t, group_network,ood_coef)

    pred_mean = torch.mean(preds_mean, axis=0)
    unc_data = torch.mean(preds_vars[:,:,[0]], axis=0)
    unc_model = torch.sum(torch.mean(preds_vars[:,:,1:], axis=0),dim=-1,keepdim=True)
    unc_Fut=torch.var(preds_mean, axis=0)+torch.sum(torch.var(preds_vars, axis=0),dim=-1,keepdim=True)
    return pred_mean,unc_model,unc_data,unc_Fut
