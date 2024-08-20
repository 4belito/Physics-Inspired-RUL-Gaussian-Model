# main
import numpy as np
import torch

# Visualization
import matplotlib.pyplot as plt 
from matplotlib.collections import PolyCollection
from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy.editor import ImageSequenceClip
from moviepy.editor import VideoClip
from IPython.display import display


# Custom modules 
from helpers import gpu2np
from method import compute_performance_distribution 

def setup_rul_video_uav():
    causes_text = ['out of power', 'pos error', 'cum time out']
    colors = ['green', 'blue', 'red']
    perform_names = ['SOC', 'POS', 'CUM']
    ax_ids = [(0, 1), (1, 0), (1, 1)]
    pred_names='acc','csv','pred','unc'
        
    perform_pred_line = dict.fromkeys(perform_names)
    performEOL={perform_name: {pred_name: None for pred_name in pred_names} for perform_name in perform_names}
    plot_predRUL={}
    plot_predRUL['pred']={pred_name+'_RUL': None for pred_name in pred_names}
    plot_predRUL['color']={pred_name+'_color': None for pred_name in pred_names}

    return causes_text, colors, perform_names, pred_names, ax_ids, perform_pred_line,performEOL,plot_predRUL

def setup_rul_video_NCMAPSS():
    causes_text = ['HPT_eff']
    colors = ['green']
    perform_names = ['HI']
    ax_ids = [0]
    pred_names='acc','csv','pred','unc'
        
    perform_pred_line = dict.fromkeys(perform_names)
    performEOL={perform_name: {pred_name: None for pred_name in pred_names} for perform_name in perform_names}
    plot_predRUL={}
    plot_predRUL['pred']={pred_name+'_RUL': None for pred_name in pred_names}
    plot_predRUL['color']={pred_name+'_color': None for pred_name in pred_names}

    return causes_text, colors, perform_names, pred_names, ax_ids, perform_pred_line,performEOL,plot_predRUL



def setup_nonlocal_var(pred_names):
    RULs={key: [] for key in pred_names}
    times= []
    EOLcauses = {key: [] for key in pred_names}
    n = 1
    return RULs,times,EOLcauses,n


def setup_plot_axes(system,conserv,ood_coef):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    #fig.suptitle(f'RUL Prediction with conserv={conserv} ood={ood_coef} (Testing UAV #{system})', fontsize=16)
    fig.suptitle(f'Aircraft Engine Remaining Useful Life prediction (N-CMAPSS dataset)', fontsize=16)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    return fig, axes

def setup_plot_perform(ax,t,thres,time_sys, b, name, system, y_lim, loc,time_unit='hours'): #time=t[:,0]
    ax.plot(t, thres, color='r', linestyle='--', label=f'threshold')
    ax.plot(time_sys, b, color='black', label='true')
    
    #Create legend (properties must agree with the corresponding plot in update function)
    ax.scatter([], [], color='green', label='observation')
    perform_pred_line, =ax.plot([], [], color='green', label='prediction')
    ax.fill_between([], [], [], color='green', alpha=0.2, label='data uncertainty')
    ax.fill_between([], [], [], color='blue', alpha=0.2, label='model uncertainty')
    ax.fill_between([], [], [], color='red', alpha=0.2, label='future uncertainty')

    # title=f'{name} Prediction + Uncertainty'
    # if system: title=title+f' (system {system})'    
    title=f'System Performance Prediction'        
    ax.set_title(title)
    ax.set_xlabel(f'time ({time_unit})')
    ax.set_ylabel(f'Health Index')
    ax.set_ylim(y_lim)
    ax.legend(loc=loc,fontsize='small')
    return perform_pred_line

def update_plot_perform(ax,mean_line,
                        t_n, b_n, t, 
                        mean, unc_Data, unc_ModelData, unc_ModelDataFut, 
                        time_est=None, b_est=None):
    ax.scatter(t_n, b_n, color='green')
    mean_line.set_data(t, mean)
    
    # if (b_est is not None) and t_n < b_est.shape[0]:
    #     ax.scatter(time_est[t_n], b_est[t_n], label='estimation', color='orange')
    for collection in ax.collections: 
        if isinstance(collection, PolyCollection): 
            collection.remove()
    ax.fill_between(t, mean - 1.96 * unc_ModelDataFut, 
                            mean - 1.96 * unc_ModelData, color='red', alpha=0.2)
    ax.fill_between(t, mean - 1.96 * unc_ModelData, 
                             mean - 1.96 * unc_Data, color='blue', alpha=0.2)
    ax.fill_between(t, mean - 1.96 * unc_Data, 
                             mean + 1.96 * unc_Data, color='green', alpha=0.2, label='data uncertainty')
    ax.fill_between(t, mean + 1.96 * unc_Data, 
                             mean + 1.96 * unc_ModelData, color='blue', alpha=0.2, label='model uncertainty')
    ax.fill_between(t, mean + 1.96 * unc_ModelData, 
                            mean + 1.96 * unc_ModelDataFut, color='red', alpha=0.2, label='future uncertainty')

def setup_plot_predRUL(ax,time_sys,true_RUL,failure_cause,causes_text,colors, loc='upper right',time_unit='hours'):
    ax.plot(time_sys,true_RUL,color='black',linestyle='dashed',label='true')
    for color,label in zip(colors, causes_text):
        ax.plot([],[],color=color,label='prediction') 
    ax.fill_between([], [], [], color='gray', alpha=0.5, label='conservatism')
    ax.fill_between([], [], [], color='gray', alpha=0.2, label='uncertainty')

    #ax.set_title(f'Prediction. EOL Cause: {failure_cause}')
    ax.set_title(f'RUL Prediction')
    ax.set_xlabel(f'time ({time_unit})')
    ax.set_ylabel(f'RUL ({time_unit})')
    ax.set_ylim(0,1.2*time_sys[-1])
    ax.legend(loc=loc)

def update_plot_predRUL(ax,t, acc_RUL,csv_RUL,pred_RUL,unc_RUL,acc_color,csv_color,pred_color,unc_color):            
    ax.plot(t, acc_RUL, color=acc_color,alpha=0.3)
    ax.plot(t, csv_RUL, color=csv_color,alpha=0.3)
    ax.plot(t, unc_RUL, color=unc_color,alpha=0.3)
    ax.plot(t, pred_RUL, color=pred_color)
    ax.fill_between(t,csv_RUL, acc_RUL, color='gray', alpha=0.5) 
    ax.fill_between(t,acc_RUL, unc_RUL, color='gray', alpha=0.2) 

def compute_uncertainties(var_Data,var_Model,var_Fut,conserv):
    std_Data = torch.sqrt(var_Data)
    var_ModelData = var_Data+var_Model
    std_ModelData = torch.sqrt(var_ModelData)
    std_ModelDataFut = torch.sqrt(var_ModelData+var_Fut)
    return (std_Data,std_ModelDataFut,torch.sqrt(std_Data**2+(1-conserv)**2*std_ModelDataFut**2)),std_ModelData    

def compute_remaining_perform_up(threshold,pred,std):
    return threshold-(pred+1.96*std)

def compute_remaining_perform_unc_up(threshold,pred,std):
    return threshold-(pred-1.96*std)

def compute_remaining_perform_down(threshold,pred,std):
    return (pred-1.96*std)-threshold

def compute_remaining_perform_unc_down(threshold,pred,std):
    return (pred+1.96*std)-threshold

def compute_performEOL(performEOL,pred_names,perform_pred,stds,
                    t,threshold,monot,max_life):   
    # compute remaining performance
    unc=stds[1]
    if monot=='+':
        remaining_performance={name:compute_remaining_perform_up(threshold,perform_pred,std) for name,std in zip(pred_names[:-1],stds)} 
        remaining_performance[pred_names[-1]]=compute_remaining_perform_unc_up(threshold,perform_pred,unc)
    elif monot=='-':
        remaining_performance={name:compute_remaining_perform_down(threshold,perform_pred,std) for name,std in zip(pred_names[:-1],stds)} 
        remaining_performance[pred_names[-1]]=compute_remaining_perform_unc_down(threshold,perform_pred,unc)
    else:
        print(f'Performance metric has an incorrect monotonicity. Must be increasing (+) or decreasing (-)')
    
    for name in pred_names:
        neg_mask=remaining_performance[name]<0
        if torch.any(neg_mask):
            neg_ind=torch.nonzero(neg_mask)[0]
            performEOL[name]=0.5*(t[neg_ind]+t[neg_ind-1])[0].item()
        else:
            performEOL[name]=max_life

def min_with_index(lst):
    # Find the minimum value and its index
    min_val, min_arg = min((val, idx) for idx, val in enumerate(lst))
    return min_val, min_arg

def highlight_preform_box(ax,color):
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(5)

def reset_preform_box(ax):
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.8)

def computeRUL(RULs,EOLcauses,performEOL,t_n):
    for name in RULs.keys():
        EOL,cause_idx=min_with_index(performEOL[perform_name][name] for perform_name in performEOL.keys())
        RULs[name].append(max(EOL-t_n,0))
        EOLcauses[name].append(cause_idx)

def fill_info_plot_predRUL(plot_predRUL,RULs,EOLcauses,pred_names,time_sys,colors,n):
    for name in pred_names:
        cause_idx=EOLcauses[name][-1]
        if n==1:
            plot_predRUL['pred'][name+'_RUL']=np.array(2*RULs[name])
            plot_predRUL['color'][name+'_color']=colors[cause_idx]
        else: 
            plot_predRUL['pred'][name+'_RUL']=np.array(RULs[name][-2:])
            plot_predRUL['color'][name+'_color']=colors[EOLcauses[name][-2]]
    if n==1:
        plot_predRUL['time'] = np.array([time_sys[n], time_sys[n]]) 
    else: 
        plot_predRUL['time'] = np.array(time_sys[n-2:n])

def update_perform_box(axes,ax_ids,n,colors,EOLcause):
    EOLidx=EOLcause[-1]
    EOLax=axes[ax_ids[EOLidx]]
    if n==1:
        highlight_preform_box(ax=EOLax,color=colors[EOLidx])
    else: 
        EOLidx_prev=EOLcause[-2]
        if EOLidx!=EOLidx_prev: 
            EOLax_prev=axes[ax_ids[EOLidx_prev]]
            reset_preform_box(ax=EOLax_prev)
            highlight_preform_box(ax=EOLax,color=colors[EOLidx])



def make_rul_video(system, t, time_sys, true_RUL, failure_cause, monot, threshold, group_networks, A, b, S, gamma, 
        y_lim, loc, time_unit,
        time_est=None, A_est=None, b_est=None, 
        n_train_sys=80, n_paths=20,ood_coef=0, 
        max_life=130, tol=1e-4, max_iter=100,
        accel=1, save=False, conserv=0.5):
    
    t_torch = torch.tensor(t, dtype=torch.float32,device=list(group_networks.values())[0].device).unsqueeze(dim=1)
    
    # Setup video
    causes_text, colors, perform_names, pred_names, ax_ids, perform_pred_line,performEOL,plot_predRUL=setup_rul_video_NCMAPSS()

    # Setup plot
    fig, axes = setup_plot_axes(system,conserv,ood_coef)
    setup_plot_predRUL(axes[1], time_sys, true_RUL, failure_cause, causes_text, colors,time_unit=time_unit)
    for i,name in enumerate(perform_names):
        perform_pred_line[name]=setup_plot_perform(axes[ax_ids[i]],t,gpu2np(threshold[name]),time_sys, b[name], name, system, y_lim[name], loc[name],time_unit)  
    
    RULs,times,EOLcauses,n = setup_nonlocal_var(pred_names)
    def make_frame(t_n):
        ood_coef
        nonlocal RULs,times,EOLcauses,n
        for i, name in enumerate(perform_names):
            An = A[name][:n]
            Sn = S[name][:n]
            bn = b[name][:n]

            # Add estimation if necessary
            if (b_est[name] is not None) and n < b_est[name].shape[0]:
                An = np.concatenate([An, A_est[name][:n]], axis=0)
                bn = np.concatenate([bn, b_est[name][:n]], axis=0)

            # perform_pred, var_Model,var_Data,var_Fut=compute_performance_distribution(t_torch,
            #                                                                     gamma=gamma[name],
            #                                                                     group_network=group_networks[name],
            #                                                                     An=An,bn=bn,Sn=Sn,
            #                                                                     n_train_sys=n_train_sys,
            #                                                                     n_paths=n_paths,ood_coef=ood_coef
            #                                                                     )
            perform_pred, var_Model,var_Data,var_Fut=compute_performance_distribution(t_torch,
                                                                                gamma=gamma[name],
                                                                                group_network=group_networks[name],
                                                                                An=An,bn=bn,Sn=Sn,
                                                                                n_train_sys=n_train_sys,
                                                                                n_paths=n_paths,ood_coef=ood_coef
                                                                                )
            # compute uncertainties
            stds,unc_ModelData = compute_uncertainties(var_Data,var_Model,var_Fut,conserv) 
            
            # Send to numpy for plotting
            perform_pred_np=gpu2np(perform_pred)
            #stds=tuple([gpu2np(std) for std in stds])
            unc_Data=gpu2np(stds[0])
            unc_ModelData=gpu2np(unc_ModelData)
            unc_ModelDataFut=gpu2np(stds[1])
                        
            compute_performEOL(performEOL[name],pred_names,perform_pred,stds,
                            t_torch,threshold[name],monot[name],max_life)
            update_plot_perform(axes[ax_ids[i]],perform_pred_line[name],
                                t_n, b[name][n-1], t, 
                                perform_pred_np, unc_Data, unc_ModelData, unc_ModelDataFut)
        computeRUL(RULs,EOLcauses,performEOL,t_n)
        fill_info_plot_predRUL(plot_predRUL,RULs,EOLcauses,pred_names,time_sys,colors,n)
        #update_perform_box(axes,ax_ids,n,colors,EOLcause=EOLcauses['pred'])
        #update_plot_predRUL(axes[0, 0], plot_predRUL['time'], **plot_predRUL['pred'],**plot_predRUL['color'])
        update_plot_predRUL(axes[1], plot_predRUL['time'], **plot_predRUL['pred'],**plot_predRUL['color'])
        frame = mplfig_to_npimage(fig)
        n += 1
        return frame

    frames = [make_frame(t) for t in time_sys.tolist()]
    animation = ImageSequenceClip(frames, fps=accel)

    if save:
        title = f"System{system}_pred"
        animation.write_videofile(f'{save}/{title}.mp4', fps=accel)
        plt.savefig(f'{save}/{title}.png')
    else:
        display(animation.ipython_display(fps=accel, loop=False, autoplay=True))

    return RULs, EOLcauses 

def plot_perform_prediction(ax, time_sys, b, t, mean, std_Data, std_ModelData, std_ModelDataFut, thres, name, system, y_lim, loc,time_unit, b_n, t_n, time_est=None, b_est=None):
    ax.set_xlabel(f'time ({time_unit})')
    ax.set_ylabel(f'{name}')
    #ax.axhline(y=thres, color='r', linestyle='--', label=f'threshold')
    ax.plot(t[:, 0], thres, color='r', linestyle='--', label=f'threshold')
    ax.plot(time_sys, b, color='black', label='true')
    ax.plot(t[:, 0], mean, color='green', label='pred')
    ax.scatter(time_sys[:t_n], b_n, label='Data', color='green')
    
    if (b_est is not None) and t_n < b_est.shape[0]:
        ax.scatter(time_est[t_n], b_est[t_n], label='estimation', color='orange')
    
    ax.fill_between(t[:, 0], mean - 1.96 * std_ModelDataFut, 
                            mean - 1.96 * std_ModelData, color='red', alpha=0.2)
    ax.fill_between(t[:, 0], mean - 1.96 * std_ModelData, 
                             mean - 1.96 * std_Data, color='blue', alpha=0.2)
    ax.fill_between(t[:, 0], mean - 1.96 * std_Data, 
                             mean + 1.96 * std_Data, color='green', alpha=0.2, label='data uncertainty')
    ax.fill_between(t[:, 0], mean + 1.96 * std_Data, 
                             mean + 1.96 * std_ModelData, color='blue', alpha=0.2, label='model uncertainty')
    ax.fill_between(t[:, 0], mean + 1.96 * std_ModelData, 
                            mean + 1.96 * std_ModelDataFut, color='red', alpha=0.2, label='future uncertainty')
    title=f'{name} Prediction + Uncertainty'
    if system: title=title+f' (system {system})'
    ax.set_title(title)
    ax.legend(loc=loc,fontsize='small')
    ax.set_ylim(y_lim)

def make_perform_video(system, group_network, thres, time_sys, A, b, S,
                    time_est=None, A_est=None, b_est=None, gamma=0.001, 
                    n_train_sys=80, n_paths=20,ood_coef=0, 
                    name='performance', end_time=80, density_time=200, frames=20, y_lim=(0,1), loc='upper left', time_unit='hours',save=False):
    frames = 20
    duration = len(time_sys) / frames
    fig, ax = plt.subplots()
    t = torch.linspace(0, end_time, density_time).reshape(density_time, 1)
    alpha_init=np.ones(n_train_sys)/n_train_sys
    def make_frame(s):
        nonlocal alpha_init
        ax.clear()
        t_n = int(s * frames) + 1
        if t_n==69: 
            print('stop')
        An=A[:t_n]
        Sn=S[:t_n]
        bn=b[:t_n]

        # Add estimation if necessary
        if (b_est is not None) and t_n < b_est.shape[0]:
            An = np.concatenate([An, A_est[:t_n]], axis=0)
            bn = np.concatenate([bn, b[:t_n]], axis=0)
        
        pred_perform, var_Model,var_Data,var_Fut=compute_performance_distribution(t,gamma,group_network,
                                                                                                An,bn,Sn,
                                                                                                n_train_sys,n_paths=n_paths,ood_coef=ood_coef)
        pred_perform=gpu2np(pred_perform)
        std_Data = gpu2np(torch.sqrt(var_Data))
        var_ModelData = var_Data+var_Model
        std_ModelData = gpu2np(torch.sqrt(var_ModelData))
        std_ModelDataFut = gpu2np(torch.sqrt(var_ModelData+var_Fut) )    

        plot_perform_prediction(ax, time_sys, b, t, pred_perform, std_Data, std_ModelData, std_ModelDataFut, thres, name, system, y_lim, loc,time_unit, bn, t_n, time_est, b_est)
        
        return mplfig_to_npimage(fig)

    animation = VideoClip(make_frame, duration=duration)
    
    if save:
        title = f"{system}_{name}.mp4"
        animation.write_videofile(title, fps=24)
    
    display(animation.ipython_display(fps=frames, loop=True, autoplay=True))
