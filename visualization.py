# mainupdate_plot_predRUL_GMM
import numpy as np
import torch
import math
# Visualization
import matplotlib.pyplot as plt 
from matplotlib.collections import PolyCollection
from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy.editor import ImageSequenceClip
from moviepy.editor import VideoClip
from IPython.display import display


# Custom modules 
from helpers import gpu2np,generate_colors
#from method_distance2mean import compute_performance_distribution,computeRUL,compute_performEOL_traj
from method import compute_performance_distribution,computeRUL,compute_performEOL_traj




def make_rul_video(unit, t_np, t_observ, true_RUL, failure_cause, 
                monot, threshold, group_networks, A, b, S, gamma, y_lim, loc,
                perform_names,causes_text, time_unit,
                n_train_unit=80, n_paths=20,lr=0, forgetting=1,
                max_life=130,
                accel=1, save=False, conserv=0.5):
    
    t_torch = torch.tensor(t_np, dtype=torch.float32,device=list(group_networks.values())[0].device).unsqueeze(1)
    n_performs=len(perform_names)
    # Setup video
    colors, pred_names, perform_pred_line,perform_pred_lines,performEOL,plot_predRUL=setup_rul_video_GMM(n_performs,perform_names) 

    # Setup plot
    title=f'RUL Prediction'
    fig, axes = setup_plot_axes(n_performs+1,title=title)
    setup_plot_predRUL(axes[0], np.concatenate([np.zeros(1),t_observ]), true_RUL, failure_cause, causes_text.values(), colors,time_unit=time_unit)
    for i,name in enumerate(perform_names):
        perform_pred_line[name],perform_pred_lines[name]=setup_plot_perform_GMM(axes[i+1],t_np,gpu2np(threshold[name]),t_observ,n_paths, b[name], name, unit, y_lim[name], loc[name],time_unit)  
    
    RULs,times,EOLcauses,n= setup_nonlocal_var(pred_names)    
    
    ## ----Method---- ##
    present_alphas,future_alphas=dict.fromkeys(perform_names),dict.fromkeys(perform_names)
    present_var_alphas,future_var_alphas=dict.fromkeys(perform_names),dict.fromkeys(perform_names)
    alpha_present=dict.fromkeys(perform_names)
    # mean_alphas=np.mean(alphas,axis=0)
    # std_alphas=np.std(alphas)
    def make_frame(t_n):
        nonlocal RULs,times,EOLcauses,n,present_alphas,future_alphas,present_var_alphas,future_var_alphas,alpha_present

        for i, name in enumerate(perform_names):
            An = A[name][:n]
            Sn = S[name][:n]
            bn = b[name][:n]
            obj_weights = np.exp(forgetting*(1-bn)) if n else None
            present_means, present_vars,future_means, future_vars,present_alphas[name],future_alphas[name],present_var_alphas[name],future_var_alphas[name],alpha_present[name]=compute_performance_distribution(t_torch,t_n,
                                                                                    gamma=gamma[name],
                                                                                    group_network=group_networks[name],
                                                                                    An=An,bn=bn,Sn=Sn,obj_weights=obj_weights,
                                                                                    n_train_unit=n_train_unit,
                                                                                    n_paths=n_paths,
                                                                                    present_alphas=present_alphas[name],future_alphas=future_alphas[name],present_var_alphas=present_var_alphas[name],future_var_alphas=future_var_alphas[name],lr=lr,alpha_present=alpha_present[name]
                                                                                    )
            
            means=torch.concat([present_means,future_means])
            present_stds=torch.sqrt(present_vars)
            future_stds=torch.sqrt(future_vars)
            stds=torch.concat([present_stds,future_stds])

            mean=torch.mean(means,axis=0)
            std=torch.mean(stds,axis=0)

            csvEOL,uncEOL=max_life,0
            for mu,sigma in zip(means,stds):            
                minEOL,maxEOL=compute_performEOL_traj(mu,sigma,t_torch,threshold[name],monot[name],max_life)
                if minEOL<csvEOL: csvEOL=minEOL
                if maxEOL>uncEOL: uncEOL=maxEOL
                        
            accEOL,_=compute_performEOL_traj(mean,std,t_torch,threshold[name],monot[name],max_life)
            performEOL[name]={'acc':accEOL,'csv':csvEOL,'unc':uncEOL}
            
            ## Plot
            mean_np=gpu2np(mean)
            std_np=gpu2np(std)
            present_means_np=gpu2np(present_means)
            present_stds_np=gpu2np(present_stds)
            future_means_np=gpu2np(future_means)
            future_stds_np=gpu2np(future_stds)
            update_plot_perform_GMM(axes[i+1],perform_pred_line[name], perform_pred_lines[name],
                        t_n, b[name][n-1], t_np,mean_np,std_np,present_means_np,present_stds_np,future_means_np,future_stds_np)
        
        computeRUL(RULs,EOLcauses,performEOL,t_n)        
        ## ------------- ##

        fill_info_plot_predRUL(plot_predRUL,RULs,EOLcauses,pred_names,t_observ,colors,n)
        update_perform_box(axes[1:],n,colors,EOLcause=EOLcauses['acc'])
        update_plot_predRUL_GMM(axes[0], plot_predRUL['time'], **plot_predRUL['pred'],**plot_predRUL['color'])
        frame = mplfig_to_npimage(fig)
        n += 1
        return frame

    frames = [make_frame(t) for t in [0]+t_observ.tolist()]
    animation = ImageSequenceClip(frames, fps=accel)

    if save:
        title = f"unit{unit}_pred"
        animation.write_videofile(f'{save}{title}.mp4', fps=accel)
        plt.savefig(f'{save}{title}.png')
    else:
        display(animation.ipython_display(fps=accel, loop=False, autoplay=True))

    return RULs, EOLcauses 


def setup_rul_video(n_perform,perform_names):
    colors = generate_colors(n_perform)
    pred_names='acc','csv','pred','unc'
        
    perform_pred_line = dict.fromkeys(perform_names)
    performEOL={perform_name: {pred_name: None for pred_name in pred_names} for perform_name in perform_names}
    plot_predRUL={}
    plot_predRUL['pred']={pred_name+'_RUL': None for pred_name in pred_names}
    plot_predRUL['color']={pred_name+'_color': None for pred_name in pred_names}

    return colors, pred_names, perform_pred_line,performEOL,plot_predRUL


def setup_rul_video_GMM(n_perform,perform_names):
    colors = generate_colors(n_perform)
    pred_names='acc','csv','unc'
        
    perform_pred_line = dict.fromkeys(perform_names)
    perform_pred_lines= dict.fromkeys(perform_names)
    performEOL={}
    plot_predRUL={}
    plot_predRUL['pred']={pred_name+'_RUL': None for pred_name in pred_names}
    plot_predRUL['color']={pred_name+'_color': None for pred_name in pred_names}

    return colors, pred_names, perform_pred_line,perform_pred_lines,performEOL,plot_predRUL

def setup_nonlocal_var(pred_names):
    RULs={key: [] for key in pred_names}
    times= []
    EOLcauses = {key: [] for key in pred_names}
    n = 0
    return RULs,times,EOLcauses,n


def setup_plot_axes(n_plots, base_size=5, title='RUL Prediction'):
    # Calculate the number of rows and columns for a nearly square grid
    rows  = int(math.sqrt(n_plots))
    cols= math.ceil(n_plots / rows)
    
    # Adjust the figure size
    figsize = (base_size * cols, base_size * rows)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    # Hide unused subplots if there are fewer plots than axes
    for i in range(n_plots, len(axes)):
        fig.delaxes(axes[i])
    
    # Set the title
    fig.suptitle(title, fontsize=16)
    
    # Adjust subplot spacing
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    return fig, axes

def setup_plot_perform(ax,t,thres,t_observ, b, name, unit, y_lim, loc,time_unit='hours'): #time=t[:,0]
    ax.plot(t, thres, color='r', linestyle='--', label=f'threshold')
    ax.plot(t_observ, b, color='black', label='true')
    
    #Create legend (properties must agree with the corresponding plot in update function)
    ax.scatter([], [], color='red', label='observations')
    perform_pred_line, =ax.plot([], [], color='green', label='prediction')
    ax.fill_between([], [], [], color='green', alpha=0.2, label='data unc')
    ax.fill_between([], [], [], color='blue', alpha=0.2, label='model unc')
    ax.fill_between([], [], [], color='red', alpha=0.2, label='future unc')

    title=f'Prediction + Uncertainty ({name})'        
    ax.set_title(title)
    ax.set_xlabel(f'time ({time_unit})')
    ax.set_ylabel(f'{name}')
    ax.set_ylim(y_lim)
    ax.legend(loc=loc,fontsize='small')
    return perform_pred_line


def setup_plot_perform_GMM(ax,t,thres,t_observ,n_paths, b, name, unit, y_lim, loc,time_unit='hours'): #time=t[:,0]
    ax.plot(t, thres, color='r', linestyle='--', label=f'threshold')
    ax.plot(t_observ, b, color='black', label='true')
    
    #Create legend (properties must agree with the corresponding plot in update function)
    ax.scatter([], [], color='red', label='observations')
    perform_pred_line, =ax.plot([], [], color='blue', label='prediction')
    perform_pred_lines=[ax.plot([], [], color='green',alpha=0.5)[0]]+[ax.plot([], [], color='green',alpha=0.5)[0] for _ in range(1,n_paths)]
    ax.plot([], [], color='orange',label='future mean',alpha=0.5)
    ax.plot([], [], color='green',label='present mean',alpha=0.5)
    ax.fill_between([], [], [], color='orange',alpha=0.2, label='future std')
    ax.fill_between([], [], [], color='green', alpha=0.2, label='present std')

    title=f'Prediction + Uncertainty ({name})'        
    ax.set_title(title)
    ax.set_xlabel(f'time ({time_unit})')
    ax.set_ylabel(f'{name}')
    ax.set_ylim(y_lim)
    ax.legend(loc=loc,fontsize='small')
    return perform_pred_line,perform_pred_lines

def update_plot_perform(ax,mean_line,
                        t_n, b_n, t, 
                        mean, unc_Data, unc_ModelData, unc_ModelDataFut, 
                        time_est=None, b_est=None):
    ax.scatter(t_n, b_n, color='orange')
    mean_line.set_data(t, mean)
    
    for collection in ax.collections: 
        if isinstance(collection, PolyCollection): 
            collection.remove()
    
    
    ax.fill_between(t, mean - 1.96 * unc_ModelDataFut, 
                            mean - 1.96 * unc_ModelData, color='red', alpha=0.2)
    ax.fill_between(t, mean - 1.96 * unc_ModelData, 
                             mean - 1.96 * unc_Data, color='blue', alpha=0.2)
    ax.fill_between(t, mean - 1.96 * unc_Data, 
                             mean + 1.96 * unc_Data, color='green', alpha=0.2, label='data unc')
    ax.fill_between(t, mean + 1.96 * unc_Data, 
                             mean + 1.96 * unc_ModelData, color='blue', alpha=0.2, label='model unc')
    ax.fill_between(t, mean + 1.96 * unc_ModelData, 
                            mean + 1.96 * unc_ModelDataFut, color='red', alpha=0.2, label='future unc')


def update_plot_perform_GMM(ax,mean_line,mean_lines,
                        t_n, b_n, t, 
                        mean,std,present_means, present_stds,future_means, future_stds):
    
    for collection in ax.collections[:]:  # The [:] creates a shallow copy of the list
        if isinstance(collection, PolyCollection):
            collection.remove()

    # Clear lines if any exist
    for line in ax.lines[2:]:
        line.remove()
    
    n=len(future_means)
    for i in range(n):
        mu = future_means[i].squeeze()
        sigma = future_stds[i].squeeze()
        
        lower_bound = mu - 1.96 * sigma
        upper_bound = mu + 1.96 * sigma
        
        ax.fill_between(t, lower_bound, upper_bound, color='orange', alpha=0.2, label='future data unc')
        mean_lines[i].set_data(t, mean)
    m=len(present_means)
    for i in range(n,n+m):
        mu = present_means[i-n].squeeze()
        sigma = present_stds[i-n].squeeze()
        
        lower_bound = mu - 1.96 * sigma
        upper_bound = mu + 1.96 * sigma
        
        ax.fill_between(t, lower_bound, upper_bound, color='green', alpha=0.2, label='present data unc')
        mean_lines[i].set_data(t, mean)
    
    mean_line.set_data(t, mean)
    lower_bound = mean - 1.96 * std
    upper_bound = mean + 1.96 * std
    ax.fill_between(t, lower_bound, upper_bound, color='blue', alpha=0.8, label='data unc')
    if t_n: ax.scatter(t_n, b_n, color='red')   


def setup_plot_predRUL(ax,t_observ,true_RUL,failure_cause,causes_text,colors, loc='upper right',time_unit='hours'):
    ax.plot(t_observ,true_RUL,color='black',linestyle='dashed',label='true')
    for color,label in zip(colors, causes_text):
        ax.plot([],[],color=color,label=label) 
    ax.fill_between([], [], [], color='gray', alpha=0.5, label='conservatism')
    ax.fill_between([], [], [], color='gray', alpha=0.2, label='uncertainty')

    ax.set_title(f'Prediction. EOL Cause: {failure_cause}')
    ax.set_xlabel(f'time ({time_unit})')
    ax.set_ylabel(f'RUL ({time_unit})')
    #ax.set_ylim(0,1.2*t_observ[-1])
    ax.legend(loc=loc)

def update_plot_predRUL(ax,t, acc_RUL,csv_RUL,pred_RUL,unc_RUL,acc_color,csv_color,pred_color,unc_color):            
    ax.plot(t, acc_RUL, color=acc_color,alpha=0.3)
    ax.plot(t, csv_RUL, color=csv_color,alpha=0.3)
    ax.plot(t, unc_RUL, color=unc_color,alpha=0.3)
    ax.plot(t, pred_RUL, color=pred_color)
    ax.fill_between(t,csv_RUL, acc_RUL, color='gray', alpha=0.5) 
    ax.fill_between(t,acc_RUL, unc_RUL, color='gray', alpha=0.2) 

def update_plot_predRUL_GMM(ax,t, acc_RUL,csv_RUL,unc_RUL,acc_color,csv_color,unc_color):            
    ax.plot(t, acc_RUL, color=acc_color,alpha=0.3)
    ax.plot(t, csv_RUL, color=csv_color,alpha=0.3)
    ax.plot(t, unc_RUL, color=unc_color,alpha=0.3)
    ax.fill_between(t,csv_RUL, acc_RUL, color='gray', alpha=0.5) 
    ax.fill_between(t,acc_RUL, unc_RUL, color='gray', alpha=0.2) 

def highlight_preform_box(ax,color):
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(5)

def reset_preform_box(ax):
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.8)

def fill_info_plot_predRUL(plot_predRUL,RULs,EOLcauses,pred_names,t_observ,colors,n):
    for name in pred_names:
        cause_idx=EOLcauses[name][-1]
        if n:
            plot_predRUL['pred'][name+'_RUL']=np.array(RULs[name][-2:])
            plot_predRUL['color'][name+'_color']=colors[EOLcauses[name][-2]]
        else: 
            plot_predRUL['pred'][name+'_RUL']=np.array(2*RULs[name])
            plot_predRUL['color'][name+'_color']=colors[cause_idx]
    t_rul=np.insert(t_observ, 0, 0)
    if n:
        plot_predRUL['time'] = t_rul[n-1:n+1]
    else: 
        plot_predRUL['time'] = np.array([0, 0]) 

def update_perform_box(axes,n,colors,EOLcause):
    EOLidx=EOLcause[-1]
    EOLax=axes[EOLidx]
    if n==1:
        EOLidx_prev=EOLcause[-1]
        if EOLidx!=EOLidx_prev: 
            EOLax_prev=axes[EOLidx_prev]
            reset_preform_box(ax=EOLax_prev)
            highlight_preform_box(ax=EOLax,color=colors[EOLidx])
    else: 
        highlight_preform_box(ax=EOLax,color=colors[EOLidx])

def plot_perform_prediction(ax, t_observ, b, t, mean, std_Data, std_ModelData, std_ModelDataFut, thres, name, unit, y_lim, loc,time_unit, b_n, t_n):
    ax.set_xlabel(f'time ({time_unit})')
    ax.set_ylabel(f'{name}')
    #ax.axhline(y=thres, color='r', linestyle='--', label=f'threshold')
    ax.plot(t, thres, color='r', linestyle='--', label=f'threshold')
    ax.plot(t_observ, b, color='black', label='true')
    ax.plot(t, mean, color='green', label='pred')
    ax.scatter(t_observ[:t_n], b_n, label='Data', color='green')
    
    ax.fill_between(t, mean - 1.96 * std_ModelDataFut, 
                            mean - 1.96 * std_ModelData, color='red', alpha=0.2)
    ax.fill_between(t, mean - 1.96 * std_ModelData, 
                             mean - 1.96 * std_Data, color='blue', alpha=0.2)
    ax.fill_between(t, mean - 1.96 * std_Data, 
                             mean + 1.96 * std_Data, color='green', alpha=0.2, label='data unc')
    ax.fill_between(t, mean + 1.96 * std_Data, 
                             mean + 1.96 * std_ModelData, color='blue', alpha=0.2, label='model unc')
    ax.fill_between(t, mean + 1.96 * std_ModelData, 
                            mean + 1.96 * std_ModelDataFut, color='red', alpha=0.2, label='future unc')
    title=f'{name} Prediction + Uncertainty'
    if unit: title=title+f' (unit {unit})'
    ax.set_title(title)
    ax.legend(loc=loc,fontsize='small')
    ax.set_ylim(y_lim)

def make_perform_video(unit, group_network, t_np,thres, t_observ, A, b, S,
                    gamma=0.001, 
                    n_train_unit=80, n_paths=20,ood_coef=0, 
                    name='performance', frames=20, y_lim=(0,1), loc='upper left', time_unit='hours',save=False):
    frames = 20
    duration = len(t_observ) / frames
    fig, ax = plt.subplots()
    t_torch = torch.tensor(t_np).unsqueeze(1)
    def make_frame(s):
        ax.clear()
        t_n = int(s * frames) + 1
        An=A[:t_n]
        Sn=S[:t_n]
        bn=b[:t_n]

        pred_perform, var_Model,var_Data,var_Fut=compute_performance_distribution(t_torch,gamma,group_network,
                                                                                                An,bn,Sn,
                                                                                                n_train_unit,n_paths=n_paths,ood_coef=ood_coef)
        pred_perform=gpu2np(pred_perform)
        std_Data = gpu2np(torch.sqrt(var_Data))
        var_ModelData = var_Data+var_Model
        std_ModelData = gpu2np(torch.sqrt(var_ModelData))
        std_ModelDataFut = gpu2np(torch.sqrt(var_ModelData+var_Fut) )    

        plot_perform_prediction(ax, t_observ, b, t_np, pred_perform, std_Data, std_ModelData, std_ModelDataFut, thres, name, unit, y_lim, loc,time_unit, bn, t_n)
        
        return mplfig_to_npimage(fig)

    animation = VideoClip(make_frame, duration=duration)
    
    if save:
        title = f"{unit}_{name}.mp4"
        animation.write_videofile(title, fps=24)
    
    display(animation.ipython_display(fps=frames, loop=True, autoplay=True))
