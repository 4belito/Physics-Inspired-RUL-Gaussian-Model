from torch import nn

class ActivationEq(nn.Module):
    def __init__(self):
        super(ActivationEq, self).__init__()

    def __eq__(self, other):
        if isinstance(other, ActivationEq):
            return type(self) == type(other)
        return False
    
    def __hash__(self):
        return hash(type(self).__name__)

class ReLU(ActivationEq):
    def __init__(self, inplace=False):
        super(ReLU, self).__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        return self.relu(x)


class Sigmoid(ActivationEq):
    def __init__(self):
        super(Sigmoid,self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(x)

class Tanh(ActivationEq):
    def __init__(self):
        super(Tanh,self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.tanh(x)
    
class Softplus(ActivationEq):
    def __init__(self):
        super(Softplus,self).__init__()
        self.softplus = nn.Softplus()

    def forward(self, x):
        return self.softplus(x)

import torch  

class LearnableSoftplus(ActivationEq):
    def __init__(self, beta_init=1.0, threshold=20):
        super(LearnableSoftplus, self).__init__()
        softinv1=torch.log(torch.exp(torch.tensor(beta_init)) - 1)
        self.positive = nn.Softplus()
        self.raw_beta = nn.Parameter(torch.tensor(softinv1))
        self.threshold = threshold

    def forward(self, x):
        beta=self.positive(self.raw_beta)
        beta_x = beta* x
        # Apply the threshold for numerical stability
        return torch.where(beta_x > self.threshold, x, 1.0 / beta * torch.log(1 + torch.exp(beta_x)))

# class LearnableSoftplus(ActivationEq):
#     def __init__(self, beta_init=1.0,beta_min=0.5,beta_max=3.0, threshold=20):
#         super(LearnableSoftplus, self).__init__()
#         softinv1=torch.log(torch.exp(torch.tensor(beta_init)) - 1)
#         self.positive = nn.Sigmoid()
#         self.raw_beta = nn.Parameter(torch.tensor(softinv1))
#         self.threshold = threshold
#         self.beta_min=beta_min
#         self.beta_max=beta_max

#     def forward(self, x):
#         beta=self.beta_min+(self.beta_max-self.beta_min)*self.positive(self.raw_beta)
#         print(beta)
#         beta_x = beta* x
#         # Apply the threshold for numerical stability
#         return torch.where(beta_x > self.threshold, x, 1.0 / beta * torch.log(1 + torch.exp(beta_x)))