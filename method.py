import torch
import numpy as np
import math
from scipy.optimize import minimize
from scipy.stats import truncnorm

# Custom Modules
from helpers import min_with_index,uncenter_var
#from helpers import euclidean_uncenter_var as uncenter_var 


def compute_performance_distribution(t,tn,gamma,group_network,An,bn,Sn,obj_weights,n_train_unit,n_paths,present_alphas,future_alphas,present_var_alphas,future_var_alphas,alpha_present,lr):
    n=len(bn)
    if n:
        pending=bn[-1] 
        pred_alphas=np.concatenate([present_alphas,future_alphas])

        #Present
        alpha_present=select_best(pred_alphas,An,bn,Sn,obj_weights) 
        #alpha_present=0.5*alpha_present_new+0.5*alpha_present
        present_var_alphas=uncenter_var(x=pred_alphas,center=alpha_present[np.newaxis,:],n=n_paths)#
        #Future
        base_prob,alpha_min=evaluate_smart_base(An,bn,Sn,obj_weights,n_train_unit,pending)
        future_var_alphas=uncenter_var(x=pred_alphas,center=np.diag(alpha_min),n=n_paths) #
    else:
        pending=1
        base_prob=np.ones(n_train_unit)/n_train_unit
        pred_alphas=np.eye(n_train_unit)[np.random.choice(n_train_unit, size=n_paths, p=base_prob)]
        
        #Present
        alpha_present=np.ones(n_train_unit)/n_train_unit
        present_var_alphas=uncenter_var(x=pred_alphas,center=alpha_present[np.newaxis,:],n=n_paths)

        #Future     
        future_var_alphas=np.ones((n_train_unit,n_train_unit))/n_train_unit
        
    
    ## Montecarlo simulation
    present_preds_mean, present_future_preds_vars,future_preds_mean, future_preds_vars,present_alphas,future_alphas=mc_simulation(group_network,t,tn,n,n_train_unit,
                                                                                                An, bn,Sn,obj_weights,
                                                                                                alpha_present,base_prob,
                                                                                                present_var_alphas, future_var_alphas,pending,n_paths,gamma) #+np.array(1e-10)
    print(f'time {n}, present alpha var: {present_var_alphas}')
    print(f'time {n}, future alpha var: {np.mean(future_var_alphas,axis=0)}')
    return present_preds_mean, present_future_preds_vars,future_preds_mean, future_preds_vars,present_alphas,future_alphas,present_var_alphas,future_var_alphas,alpha_present


## Clean this
def mc_simulation(group_network,t,tn,n,n_train_unit,An, bn,Sn,obj_weights,present_alpha,future_prob,present_var,future_var,pending,n_paths, gamma):
    n_sampled=0
    present_preds_mean,future_preds_mean=[],[]
    present_preds_vars,future_preds_vars=[],[]
    present_alphas,future_alphas=[],[]

    mean_future_var=np.mean(future_var,axis=1)
    mean_present_var=np.mean(present_var)
    future_rate=pending*np.mean(mean_future_var)/(np.mean(mean_future_var)+mean_present_var)
    present_rate= 1-future_rate# #try  2: pending only
    future_std=np.sqrt(future_var)
    present_std=np.sqrt(present_var)
    future_belief_size=np.clip(future_std,0,1)#(1+pending)*future_std/(present_std+future_std)
    present_belief_size=np.clip(present_std,0,1)
    while n_sampled<n_paths:
        alphas=sample_present(present_alpha,math.ceil((n_paths-n_sampled)*present_rate), n_train_unit,present_belief_size)
        if len(alphas) and n: 
            alphas=adjust_present(An, bn,Sn, gamma*present_var,pending,alphas,obj_weights) 
        alphas_torch=torch.tensor(alphas, dtype=torch.float32,device=group_network.device)     
        preds_mean, preds_vars= trajectory_evaluation(group_network,t,alphas_torch)
        preds_vars=preds_vars[...,0]
        preds_mean=preds_mean[...,0]
        # Prune the no_monotonic simulations
        stds=torch.sqrt(preds_vars)
        mask= is_monotonic_decreasing((preds_mean+1.96*stds)[:,t[:,0]>=tn]) #>=tn
        #mask= torch.tensor([True for _ in preds_mean]) 
        if any(mask):
            n_sampled+=sum(mask).item()        
            present_preds_mean.append(preds_mean[mask])
            present_preds_vars.append(preds_vars[mask])
            present_alphas.append(alphas_torch[mask])
        
        # Future
        alphas,center_choice=sample_future(future_prob,math.floor((n_paths-n_sampled)*future_rate), n_train_unit,future_belief_size)
        if len(alphas) and n: 
            alphas=adjust_future(An, bn,Sn, gamma*future_var[center_choice],pending,alphas,obj_weights) 
        alphas_torch=torch.tensor(alphas, dtype=torch.float32,device=group_network.device)     
        preds_mean, preds_vars= trajectory_evaluation(group_network,t,alphas_torch)
        # Prune the no_monotonic simulations
        preds_vars=preds_vars[...,0]
        preds_mean=preds_mean[...,0]
        stds=torch.sqrt(preds_vars)
        mask= is_monotonic_decreasing((preds_mean+1.96*stds)[:,t[:,0]>=tn]) 
        #mask= torch.tensor([True for _ in preds_mean]) 
        if any(mask):
            n_sampled+=sum(mask).item()        
            future_preds_mean.append(preds_mean[mask])
            future_preds_vars.append(preds_vars[mask])
            future_alphas.append(alphas_torch[mask])
    time_len=len(t)
    present_preds_mean=torch.concat(present_preds_mean) if len(present_preds_mean) else torch.empty((0,time_len),device=group_network.device)
    present_preds_vars=torch.concat(present_preds_vars) if len(present_preds_vars) else torch.empty((0,time_len),device=group_network.device)
    present_alphas=torch.concat(present_alphas) if len(present_alphas) else torch.empty((0,n_train_unit),device=group_network.device)
    future_preds_mean=torch.concat(future_preds_mean) if len(future_preds_mean) else torch.empty((0,time_len),device=group_network.device)
    future_preds_vars=torch.concat(future_preds_vars) if len(future_preds_vars) else torch.empty((0,time_len),device=group_network.device)
    future_alphas=torch.concat(future_alphas) if len(future_alphas) else torch.empty((0,n_train_unit),device=group_network.device)
    return present_preds_mean,present_preds_vars,future_preds_mean,future_preds_vars,present_alphas.cpu().numpy(),future_alphas.cpu().numpy()

## Selection
def select_best(alphas,An,bn,Sn,obj_weights):
    scores=[gaussian_density(alpha,An, bn,Sn,obj_weights) for alpha in alphas]
    min_index = scores.index(min(scores))
    best_alpha=alphas[min_index,:]
    return best_alpha


## Objectives functions
def gaussian_density(alpha, A, b, S,weights):
    return np.average((A.dot(alpha) - b)**2/S.dot(alpha**2)+np.log(S.dot(alpha**2)),weights=weights)#

def convex_objective(alpha, A, b, S,gamma,alpha0,pending,weights):#,progress
    #return gaussian_density(alpha, A, b, S)+gamma*np.mean((alpha-alpha0)**2)
    return (1-pending)*gaussian_density(alpha, A, b, S,weights)+pending*np.mean(gamma*(alpha-alpha0)**2)

## Evaluation
def evaluate_smart_base(An,bn,Sn,obj_weights,n_train_unit,pending):
    constr = [] #{'type': 'ineq', 'fun': pos_constraint},{'type': 'ineq', 'fun': p_norm_constraint}
    obj_weights = np.exp(30*(1-bn)) 
    alphas_min=np.concatenate([minimize(gaussian_density, x0=1,  args=(An[:,[i]], bn,Sn[:,[i]],obj_weights),constraints=constr).x for i in range(n_train_unit)])
    scores=np.array([gaussian_density(alpha,An[:,i], bn,Sn[:,i],obj_weights) for i,alpha in enumerate(alphas_min)])
    e_values = np.exp(-(scores-np.min(scores))**(1-pending))  
    prob = e_values / np.sum(e_values)
    return prob,alphas_min

def evaluate_base(An,bn,Sn,obj_weights,n_train_unit,pending):
    progress=1-pending
    scores=np.array([gaussian_density(alpha,An, bn,Sn,obj_weights) for alpha in np.eye(n_train_unit)])
    e_values = np.exp(-(np.max(scores)-scores)**progress)  # this gives more probability to the worst bases
    prob = e_values / np.sum(e_values)
    return prob



## sample methods
def sample_present(best_alpha,n_samples, n_train_unit,std):
    present_alphas=np.tile(best_alpha,(n_samples,1))+center_mean_noise(n_train_unit,n_samples,std,abs=False)
    return present_alphas

def sample_future(base_prob,n_samples, n_train_unit,std):
    center_choice=np.random.choice(n_train_unit, size=n_samples, p=base_prob)
    future_alphas = np.eye(n_train_unit)[center_choice] #
    std=std[center_choice]
    future_alphas+=center_mean_noise(n_train_unit,n_samples,std,abs=False)#inertia=inertia,
    return future_alphas,center_choice

# Add Noise
def center_mean_noise(n_vectors,n_samples,std,abs=False):
    if abs: 
        eps=truncnorm.rvs(0, np.inf, loc=0, scale=1, size=(n_samples,n_vectors))*std
    else:
        eps=np.random.normal(0, std, (n_samples,n_vectors))*std
    return eps-np.mean(eps,axis=-1,keepdims=True)

## Simulation
def adjust(An, bn,Sn, gamma,pending,alphas,weights):
    constr = [] #{'type': 'ineq', 'fun': pos_constraint},{'type': 'ineq', 'fun': p_norm_constraint}
    alphas=[minimize(convex_objective, x0=alpha,  args=(An, bn,Sn, gamma,alpha,pending,weights),constraints=constr).x for alpha in alphas]
    return np.stack(alphas)

def adjust_present(An, bn,Sn, gamma,pending,alphas,weights):
    constr = [] #{'type': 'ineq', 'fun': pos_constraint},{'type': 'ineq', 'fun': p_norm_constraint}
    alphas=[minimize(convex_objective, x0=alpha,  args=(An, bn,Sn, gamma,alpha,pending,weights),constraints=constr).x for alpha in alphas]
    return np.stack(alphas)


def adjust_future(An, bn,Sn, gammas,pending,alphas,weights):
    constr = [] #{'type': 'ineq', 'fun': pos_constraint},{'type': 'ineq', 'fun': p_norm_constraint}
    alphas=[minimize(convex_objective, x0=alpha,  args=(An, bn,Sn, gamma,alpha,pending,weights),constraints=constr).x for alpha,gamma in zip(alphas,gammas)]
    return np.stack(alphas)

def trajectory_evaluation(group_network,t,alphas_sample):    
    preds_mean,preds_vars=group_network.forward_gaussian_lc(t,alphas_sample)
    return preds_mean,preds_vars

def is_monotonic_decreasing(fut_prediction):
    diffs = torch.diff(fut_prediction, dim=1)
    return torch.all(diffs <= 0, dim=1)



# def compute_uncertainties(var_Data,var_Model,var_Fut,conserv):
#     std_Data = torch.sqrt(var_Data)
#     var_ModelData = var_Data+var_Model
#     std_ModelData = torch.sqrt(var_ModelData)
#     std_ModelDataFut = torch.sqrt(var_ModelData+var_Fut)
#     return (std_Data,std_ModelDataFut,torch.sqrt(conserv**2*std_Data**2+(1-conserv)**2*std_ModelDataFut**2)),std_ModelData   



# def compute_performEOL(performEOL,pred_names,perform_pred,stds,
#                     t,threshold,monot,max_life):   
#     # compute remaining performance
#     unc=stds[1]
#     if monot=='+':
#         remaining_performance={name:compute_remaining_perform_up(threshold,perform_pred,std) for name,std in zip(pred_names[:-1],stds)} 
#         remaining_performance[pred_names[-1]]=compute_remaining_perform_unc_up(threshold,perform_pred,unc)
#     elif monot=='-':
#         remaining_performance={name:compute_remaining_perform_down(threshold,perform_pred,std) for name,std in zip(pred_names[:-1],stds)} 
#         remaining_performance[pred_names[-1]]=compute_remaining_perform_unc_down(threshold,perform_pred,unc)
#     else:
#         print(f'Performance metric has an incorrect monotonicity. Must be increasing (+) or decreasing (-)')
    
#     for name in pred_names:
#         neg_mask=remaining_performance[name]<0
#         if torch.any(neg_mask):
#             neg_ind=torch.nonzero(neg_mask)[0]
#             performEOL[name]=0.5*(t[neg_ind]+t[neg_ind-1])[0].item()
#         else:
#             performEOL[name]=max_life


##### Used externally ###############
## Preprocessing data for efficiency
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


def computeRUL(RULs,EOLcauses,performEOL,t_n):
    for name in RULs.keys():
        EOL,cause_idx=min_with_index(performEOL[perform_name][name] for perform_name in performEOL.keys())
        RULs[name].append(max(EOL-t_n,0))
        EOLcauses[name].append(cause_idx)


def compute_performEOL_traj(mu,std,t,threshold,monot,max_life):  

    # compute remaining performance
    if monot=='+':
        remaining_performance=compute_remaining_perform_up(threshold,mu,std)  
        remaining_performance_unc=compute_remaining_perform_unc_up(threshold,mu,std)
    elif monot=='-':
        remaining_performance=compute_remaining_perform_down(threshold,mu,std)
        remaining_performance_unc=compute_remaining_perform_unc_down(threshold,mu,std)
    else:
        print(f'Performance metric has an incorrect monotonicity. Must be increasing (+) or decreasing (-)')
    
    performEOL=[]
    for remain in [remaining_performance,remaining_performance_unc]:
        neg_mask=remain<0
        if torch.any(neg_mask):
            neg_ind=torch.nonzero(neg_mask)[0]
            #performEOL.append(0.5*(t[neg_ind]+t[neg_ind-1])[0].item())
            performEOL.append(torch.floor(0.5*(t[neg_ind]+t[neg_ind-1])[0]).item())
        else:
            performEOL.append(max_life)
    
    return tuple(performEOL)



def compute_remaining_perform_up(threshold,pred,std):
    return threshold-(pred+1.96*std)

def compute_remaining_perform_unc_up(threshold,pred,std):
    return threshold-(pred-1.96*std)

def compute_remaining_perform_down(threshold,pred,std):
    return (pred-1.96*std)-threshold

def compute_remaining_perform_unc_down(threshold,pred,std):
    return (pred+1.96*std)-threshold



