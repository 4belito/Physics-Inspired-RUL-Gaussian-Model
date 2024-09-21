import torch
from torch import nn
import numpy as np
from helpers import high_derivate,derivate_vector,gpu2np


# Auxiliary Net Constructors
class PolynomialNet(nn.Module):
    def __init__(self, coefficients):
        super().__init__()
        self.coefficients = coefficients[::-1]  # Reverse the coefficients

    def forward(self, x):
        y = torch.zeros_like(x)
        for i, coef in enumerate(self.coefficients):
            y += coef * x**i
        return y
    
class PolyTaylorNet(nn.Module):
    def __init__(self, derivatives,a):
        super().__init__()
        self.derivatives = derivatives
        self.n_coef=len(derivatives)
        self.a=a
        self.factorials=self.compute_factorials()

    def compute_factorials(self):
        factorials = self.n_coef*[1]
        for n in range(1, self.n_coef):
            factorials[n] = factorials[n - 1] * n
        return factorials
    
    def forward(self, x):
        y = torch.zeros_like(x)

        for i, der in enumerate(self.derivatives):
            y += der * (x-self.a)**i/self.factorials[i]
        return y
    
class ConcatNets(nn.Module):
    def __init__(self, net1, net2, limit):
        super().__init__()
        self.net1 = net1
        self.net2 = net2
        self.limit = limit

    def forward(self, x):
        mask = (x[:,0] <= self.limit)
        out1=self.net1(x[mask])
        out2=self.net2(x[~mask])
        out=torch.empty([x.size(0),out1.size(1)],device=x.device)
        out[mask] = out1
        out[~mask] = out2
        return out

class ZeroNet(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, x):
        batch_size = x.size(0)  # Keep the batch size from the input
        return torch.zeros(batch_size, self.output_dim, device=x.device)

class AddNets(nn.Module):
    def __init__(self, net1, net2):
        super().__init__()
        self.net1 = net1
        self.net2 = net2

    def forward(self, x):
        return self.net1(x) + self.net2(x)

class VectorNet(nn.Module):
    def __init__(self, networks):
        super().__init__()
        self.networks = nn.ModuleList(networks)

    def forward(self, x):
        outputs = [net(x) for net in self.networks]
        combined_output = torch.cat(outputs, dim=1)  # Concatenate along the feature dimension\n
        return combined_output 


## Watercan Model

#Individual hole net
class MonotLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, positive_outs=None,monot_outs=None):
        super().__init__()
        assert positive_outs is None or len(positive_outs)==out_features, 'The positive_out argument has to be a boolean list with len out_features or None'
        assert monot_outs is None or len(monot_outs)==out_features, 'The monot_out argument has to be a boolean list with len out_features or None'
        softinv1=0.541324854612918 #~torch.log(torch.exp(torch.tensor(1.0)) - 1)
        self.weight = nn.Parameter(torch.randn(out_features, in_features)+softinv1)
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        self.positive = nn.Softplus()
        positive_outs = torch.tensor(positive_outs) if positive_outs else torch.tensor(out_features * [False])
        monot_outs=torch.tensor(monot_outs, dtype=torch.bool).unsqueeze(1) if monot_outs else torch.tensor(out_features * [True], dtype=torch.bool).unsqueeze(1)  
        self.register_buffer('positive_outs', positive_outs)
        self.register_buffer('monot_outs', monot_outs)
        
    def load_avg_net(self, nets):
        weights = [[param.clone().detach() for param in net.parameters()] for net in nets]
        weights_avg = []
        for param_group in zip(*weights):
            avg_param = torch.mean(torch.stack(param_group), dim=0)
            weights_avg.append(avg_param)
        for param_self, avg_param in zip(self.parameters(), weights_avg):
            param_self.data.copy_(avg_param)

    def forward(self, x):
        sign_weight =  torch.where(self.monot_outs, self.positive(self.weight), -self.positive(self.weight)) 
        out=nn.functional.linear(x, sign_weight, self.bias)
        out=torch.where(self.positive_outs, self.positive(out), out)
        return out

class MonotIsoLinear(nn.Module):
    def __init__(self, features, bias=True, positive_outs=None,monot_outs=None):
        super().__init__()
        assert positive_outs is None or len(positive_outs)==features, 'The positive_out parameters has to be a boolean list with len our_features or None'
        assert monot_outs is None or len(monot_outs)==features, 'The monot_out parameters has to be a boolean list with len our_features or None'
        self.weight = nn.Parameter(torch.randn(features))
        if bias:
            self.bias = nn.Parameter(torch.randn(features))
        else:
            self.register_parameter('bias', None)
        self.positive = nn.Softplus()
        positive_outs = torch.tensor(positive_outs) if positive_outs else torch.tensor(features * [False])
        monot_outs=torch.tensor(monot_outs, dtype=torch.bool) if monot_outs else torch.tensor(features * [True], dtype=torch.bool) 
        self.register_buffer('positive_outs', positive_outs)
        self.register_buffer('monot_outs', monot_outs)

    def load_avg_net(self, nets):
        weights = [[param.clone().detach() for param in net.parameters()] for net in nets]
        weights_avg = []
        for param_group in zip(*weights):
            avg_param = torch.mean(torch.stack(param_group), dim=0)
            weights_avg.append(avg_param)
        for param_self, avg_param in zip(self.parameters(), weights_avg):
            param_self.data.copy_(avg_param)

    def forward(self, x):
        sign_weight =  torch.where(self.monot_outs, self.positive(self.weight), -self.positive(self.weight)) #
        out=x*sign_weight+ self.bias
        out=torch.where(self.positive_outs, self.positive(out), out)
        return out
    
## Group all the holes
class Holes(nn.Module):
    def __init__(self,input_dim,out_dim,n_holes=80,positive_outs=None,monot_outs=None,isolate=True):
        super().__init__()
        if isolate:
            assert input_dim==out_dim, 'If you isolate weights, input and output dim must be the same'
        # Create structure of the network
        self.holes=nn.ModuleList()
        self.n_holes=n_holes
        self.input_dim=input_dim
        self.out_dim=out_dim
        self.positive_outs=positive_outs
        self.monot_outs=monot_outs
        self.isolate=isolate
        for _ in range(n_holes):
            if isolate:
                hole=MonotIsoLinear(input_dim,positive_outs=positive_outs,monot_outs=monot_outs)
            else:
                hole=MonotLinear(input_dim,out_dim,positive_outs=positive_outs,monot_outs=monot_outs)
            self.holes.append(hole)

    def get_weights_std(self):
        weights = [hole.parameters() for hole in self.holes] 
        weights_std = []
        for param_group in zip(*weights):
            std_param = torch.std(torch.stack(param_group), dim=0)
            weights_std.append(std_param)
        return weights_std

    def get_weights_mean(self):
        weights = [hole.parameters() for hole in self.holes]  #[[param for param in hole.parameters()] for hole in self.holes]
        weights_mean = []
        for param_group in zip(*weights):
            mean_param = torch.mean(torch.stack(param_group), dim=0)
            weights_mean.append(mean_param)
        return weights_mean

    def get_weights_zscores(self):
        weights = [torch.cat([param.view(-1) for param in hole.parameters()]) for hole in self.holes]
        weights = torch.stack(weights)  # Shape: (num_holes, num_params)
        mean_param = torch.mean(weights, dim=0)
        std_param = torch.std(weights, dim=0)

        # Compute the z-score
        weights_zscore = (weights - mean_param) / std_param
        return weights_zscore
    
    def get_weight_zscore_norm(self,p=2):
        return torch.norm(self.get_weights_zscores(),p=p,dim=0)

    def get_zscore_norm(self,p=2):
        return torch.norm(self.get_weights_zscores(),p=p) 
    
    def get_sub_holes(self,holes_list):
        sub_holes=Holes(self.input_dim,self.out_dim,n_holes=len(holes_list),positive_outs=self.positive_outs,monot_outs=self.monot_outs,isolate=self.isolate)
        for i,hole_id in enumerate(holes_list):
            sub_holes.holes[i].load_state_dict(self.holes[hole_id].state_dict()) 
        return sub_holes

    def forward(self, x):
        outs=[]
        for i,hole in enumerate(self.holes):
            outs.append(hole(x[i]))
        return outs

#Latent Funciton Net
class LatentF(nn.Module): 
    def __init__(self, hidden_sizes=[3],channels_out=2,activation=nn.Softplus()):
        super().__init__()
        # save arguments
        self.hidden_sizes=hidden_sizes
        self.channels_out=channels_out 
        self.activation=activation       

        # Create structure of the network
        self.no_linear= nn.ModuleList()
        self.linear= nn.ModuleList()
        hidden_sizes=[1]+hidden_sizes+[channels_out]
        for i in range(1,len(hidden_sizes)):
            no_linear=nn.Sequential(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]),activation)
            self.no_linear.append(no_linear)
            self.linear.append(nn.Linear(hidden_sizes[i-1],1))        

    def forward(self, x):
        for linear,no_linear in zip(self.linear,self.no_linear):
            no_linear_x=no_linear(x)
            x = linear(x)*torch.ones_like(no_linear_x)+no_linear_x
        return x
    
## Build the  model
class WaterCan(nn.Module):
    def __init__(self,hidden_sizes,channels_out=2,n_holes=1,monot='+',isolate_outs=True,activation=nn.Softplus()):
        super().__init__()
        assert (isolate_outs and channels_out==2) or (not isolate_outs), 'If isolate outs, then channels out must be 2'
        # Properties
        self.hidden_sizes=hidden_sizes
        self.channels_out=channels_out 
        self.n_holes=n_holes
        self.monot=monot
        self.isolate_outs=isolate_outs
        self.activation=activation
        
        ## Structure
        self.holes_in=Holes(1,1,n_holes=n_holes)
        self.latentf=LatentF(hidden_sizes,channels_out,activation) 
        self.monot_outs=[True,True] if monot=='+' else [False,True]
        self.holes_out=Holes(channels_out,2,n_holes=n_holes,positive_outs=[False,True],monot_outs=self.monot_outs,isolate=isolate_outs)

        # Initial state
        self.latentf_ext = None
        self.latentf_out = self.latentf
        self._device = 'cpu'

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self._device = device

    def to(self, *args, **kwargs):
        result = super(WaterCan,self).to(*args, **kwargs)
        if isinstance(args[0], torch.device):
            self.device = args[0]
        elif isinstance(args[0], str):
            self.device = torch.device(args[0])
        return result

    def contract(self):
        self.latentf_out=self.latentf
    
    def extend(self):
        self.latentf_out=self.latentf_ext

    def set_extension(self,xs_train,n_points,poly_deg):
        xs=self.holes_in(xs_train)
        x_full = torch.cat(xs, dim=0)
        y=self.latentf(x_full)
        x, indices = torch.sort(x_full,dim=0)
        y=y[indices,:]
        self.limit=x[-1].item().requires_grad()
        poly_exts=nn.ModuleList()
        x=gpu2np(x)[-n_points:]
        y=gpu2np(y)[-n_points:]
        for channel in range(self.channels_out):
            coefficients = np.polyfit(x,y[:,channel], poly_deg)
            poly_exts.append(PolynomialNet(coefficients))
        self.latentf_ext=ConcatNets(self.latentf,VectorNet(poly_exts),limit=self.limit)

    def set_taylor_extension(self,xs_train,poly_deg):
        xs=self.holes_in(xs_train)
        x_full = torch.cat(xs, dim=0)
        x_limit,_ = torch.max(x_full,dim=0,keepdim=True)
        x_limit.requires_grad_()
        y_limit=self.latentf(x_limit)
        poly_exts=nn.ModuleList()
        for channel in range(self.channels_out):
            derivatives = high_derivate(y_limit[:,[channel]],x_limit,ord=poly_deg,cumul=True)
            derivatives =[der.squeeze() for der in derivatives]
            poly_exts.append(PolyTaylorNet(derivatives,x_limit))
        self.limit=x_limit.squeeze()
        self.latentf_ext=ConcatNets(self.latentf,VectorNet(poly_exts),limit=self.limit)
        

    def get_latentf_avg(self):
        holes_in_avg=MonotLinear(1,1) 
        if self.isolate_outs:
            holes_out_avg=MonotIsoLinear(self.channels_out,positive_outs=[False,True],monot_outs=self.monot_outs)
        else:
            holes_out_avg=MonotLinear(self.channels_out,2,positive_outs=[False,True],monot_outs=self.monot_outs)
        holes_in_avg.load_avg_net(self.holes_in.holes)
        holes_out_avg.load_avg_net(self.holes_out.holes)

        latentf_avg=nn.Sequential(holes_in_avg,self.latentf_out,holes_out_avg).to(self.device)

        return latentf_avg
    
    def get_sub_watercan(self,holes_list):
        sub_watercan=WaterCan(hidden_sizes=self.hidden_sizes,
                            channels_out=self.channels_out,
                            n_holes=len(holes_list),
                            monot=self.monot,
                            isolate_outs=self.isolate_outs,
                            activation=self.activation)
        sub_watercan.to(self.device)
        sub_watercan.holes_in.load_state_dict(self.holes_in.get_sub_holes(holes_list).state_dict())
        sub_watercan.latentf.load_state_dict(self.latentf.state_dict())
        sub_watercan.holes_out.load_state_dict(self.holes_out.get_sub_holes(holes_list).state_dict())
        return sub_watercan

    def get_holes_zscore_norm(self,p=2):
        return torch.stack((self.holes_in.get_zscore_norm(p=p),self.holes_out.get_zscore_norm(p=p)))
    
    def get_weight_zscore_norm(self,p=2):
        return torch.cat((self.holes_in.get_weight_zscore_norm(p=p),self.holes_out.get_weight_zscore_norm(p=p)))

    def get_trajectory(self,i):
        hole_in=MonotLinear(1,1) 
        if self.isolate:
            hole_out=MonotIsoLinear(self.channels_out,positive_outs=[False,True],monot_outs=self.monot_outs)
        else:
            hole_out=MonotLinear(self.channels_out,2,positive_outs=[False,True],monot_outs=self.monot_outs)
        hole_in.load_state_dict(self.holes_in.holes[i].parameters().clone().detach())
        hole_out.load_state_dict(self.holes_in.holes[i].parameters().clone().detach())

        trajectory_net=nn.Sequential(hole_in,self.latentf,hole_out).to(self.device)
        return trajectory_net  



    def forward_fast(self,xs,split_sizes,der_order=0,get_ext=None):
        xs=self.holes_in(xs)
        x_full = torch.cat(xs, dim=0)
        if der_order:
            self.x_ext=get_ext(x_full,device=self.device)
            x_full_ext = torch.cat([x_full,self.x_ext], dim=0)
            act_full=self.latentf_out(x_full_ext)
        else:
            act_full=self.latentf_out(x_full)
        xs = torch.split(act_full, split_sizes)
        if der_order: 
            act_ext=xs[-1]
            xs=xs[:-1] 
            act_ext_list=list(torch.split(act_ext, 1, dim=-1))
            act_d = derivate_vector(act_ext_list,self.x_ext)
            if der_order>1:
                act_dd = derivate_vector(act_d,self.x_ext)
                if der_order == 3:
                    act_ddd = derivate_vector(act_dd,self.x_ext)
        
        outs=self.holes_out(xs)
        match der_order:
            case 0:
                act_der=None
            case 1:
                act_der=torch.cat(act_d,dim=-1)
            case 2:
                act_der=torch.cat(act_d,dim=-1),torch.cat(act_dd,dim=-1)
            case 3:
                act_der=torch.cat(act_d,dim=-1),torch.cat(act_ddd,dim=-1)

        return outs,act_der

    def forward(self, xs):
        split_sizes = [x.size(0) for x in xs]        
        outs,_=self.forward_fast(xs,split_sizes)
        return outs


## Group Distribution(to account for model uncertainty)
# We train several watercan models and then GroupWatercan class group them in only one (mean)model that also accounts for model uncertainty.
def distribution_out(mean,var,vars,out_dist,out_std):
    if out_dist:
        if out_std:
            std=torch.sqrt(var)
            return torch.cat([mean,std],dim=1)
        else:
            return torch.cat([mean,var],dim=1)
    else:
        data_var=torch.mean(vars,dim=0)
        model_var=var
        if out_std:
            data_std=torch.sqrt(data_var)
            model_std=torch.sqrt(var)
            return mean,torch.cat([data_std,model_std],dim=1)
        else:
            return mean,torch.cat([data_var,model_var],dim=1)

# def group_distribution(group_output,out_dist=True,out_std=True): 
#     mean=torch.mean(group_output[:,:,[0]],dim=0)
#     mean_var=torch.mean(group_output[:,:,[1]]**2,dim=0)
#     var_group=torch.var(group_output,dim=0)
#     vars=torch.cat([mean_var,var_group],dim=1)
#     return distribution_out(mean,vars,out_dist,out_std)


def GMM_distribution(output,out_dist=True,out_std=True): # Gaussian mixture model
    means=output[:,:,[0]]
    vars=output[:,:,[1]]**2
    mean=torch.mean(means,dim=0)
    var=torch.mean(vars+(means-mean)**2,dim=0)
    return distribution_out(mean,var,vars,out_dist,out_std)

class GroupDistributionNet(nn.Module):
    def __init__(self, networks):
        super().__init__()
        self.networks = nn.ModuleList(networks)

    def forward(self, x):
        outputs =torch.stack( [net(x) for net in self.networks])
        return GMM_distribution(outputs)

class GroupWaterCan(nn.Module):
    def __init__(self,configs):#,std_bound=float('inf')
        super().__init__()
        n_holes = configs[0]['n_holes']
        n_models = len(configs)
        assert all(configs[i]['n_holes'] == n_holes for i in range(1,n_models)), 'All watercans must have the same number of holes'

        watercans=[]
        for config in configs:
            watercan=WaterCan(**config)
            watercans.append(watercan)

        self.watercans=nn.ModuleList(watercans)
        self.configs=configs
        self.n_holes = n_holes
        self.n_models = n_models 
        self._device = 'cpu'
        #self.std_bound=std_bound

    def load_watercans(self,state_dicts):
        for watercan,state_dict in zip(self.watercans,state_dicts):
            watercan.load_state_dict(state_dict)

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self._device = device

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        for w in self.watercans: w.to(*args, **kwargs)
        if isinstance(args[0], torch.device):
            self.device = args[0]
        elif isinstance(args[0], str):
            self.device = torch.device(args[0])
        return result

    def contract(self):
        for watercan in self.watercans: watercan.contract()
    
    def extend(self):
        for watercan in self.watercans: watercan.extend()

    def set_extension(self,xs_train,n_points,poly_deg):
        # Find the polynomial coefficients
        for watercan in self.watercans: watercan.set_extension(xs_train,n_points,poly_deg)

    def set_taylor_extension(self,xs_train,poly_deg):
        for watercan in self.watercans: watercan.set_taylor_extension(xs_train,poly_deg)
        
    def get_latentf_avg(self):
        latentf_avg=[]
        for watercan in self.watercans: latentf_avg.append(watercan.get_latentf_avg())
        return GroupDistributionNet(latentf_avg)
    

    def get_sub_watercan(self,holes_list):
        sub_configs=self.configs
        n_holes=len(holes_list)
        sub_state_dicts=[]
        for i,watercan in enumerate(self.watercans): 
            sub_watercan=watercan.get_sub_watercan(holes_list)
            sub_configs[i]['n_holes']=n_holes
            sub_state_dicts.append(sub_watercan.state_dict())
        sub_model=GroupWaterCan(sub_configs)
        sub_model.load_watercans(sub_state_dicts)
        sub_model.to(self.device)
        return sub_model

    def forward_lc(self,x, alpha,out_dist=True,out_std=True):
        means,vars=zip(*self.forward_full(self.n_holes*[x],out_dist=False,out_std=False))        
        # Compute the weighted sum of means and stds
        mean_lc = torch.einsum('i,ijk->jk', alpha, torch.stack(means))
        vars_lc = torch.einsum('i,ijk->jk', alpha**2, torch.stack(vars)) 
        return distribution_out(mean_lc,vars_lc,out_dist,out_std)
    
    def forward_gaussian_lc(self,x, alpha):
        means,vars=zip(*self.forward_full(self.n_holes*[x],out_dist=False,out_std=False))        
        # Compute the weighted sum of means and stds
        mean_lc = torch.einsum('ai,ijk->ajk', alpha, torch.stack(means))
        vars_lc = torch.einsum('ai,ijk->ajk', alpha**2, torch.stack(vars)) 
        return mean_lc,vars_lc
    
    def append(self,watercan):
        assert watercan.device == self.watercans[0].device, 'the added watercan must be in the same device as the GroupWaterCan'
        assert watercan.n_holes == self.watercans[0].n_holes, 'the added watercan must must have the same number of holes as the GroupWaterCan'
        self.watercans.append(watercan)
        self.n_models+=1
    
    # def forward_full(self,xs,out_dist,out_std):
    #     with torch.no_grad():
    #         split_sizes = [x.size(0) for x in xs]  
    #         preds= [watercan.forward_fast(xs,split_sizes)[0] for watercan in self.watercans]
    #         trajs_preds =[torch.stack(traj_preds) for traj_preds in zip(*preds)]
    #         out=[GMM_distribution(trajs_preds[hole],out_dist,out_std) for hole in range(self.n_holes)]
    #     return out
    

    def forward_full(self,xs,out_dist,out_std):
        with torch.no_grad():
            split_sizes = [x.size(0) for x in xs]  
            preds= [watercan.forward_fast(xs,split_sizes)[0] for watercan in self.watercans]
            trajs_preds =[torch.stack(traj_preds) for traj_preds in zip(*preds)]
            #for traj in trajs_preds: traj[...,1]=torch.clip(traj[...,1],0,self.std_bound) 
            out=[GMM_distribution(trajs_preds[hole],out_dist,out_std) for hole in range(self.n_holes)]
        return out

    

    def forward_single(self,x):
        return torch.stack(self(self.n_holes*[x]),dim=1)
    
    def forward(self,xs):
        return self.forward_full(xs,out_dist=True,out_std=True)
    

