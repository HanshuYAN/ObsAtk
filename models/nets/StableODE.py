import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd.functional import jacobian

sys.path.append('..')
import torchdiffeq._impl.odeint as odeint

# Stable Neural ODE
def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class ConcatConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )
    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)

           
class ODEfunc(nn.Module):
    def __init__(self, depth=17, n_filters=64, kernel_size=3, img_channels=1):
        super(ODEfunc, self).__init__()
        layers = [
            nn.Conv2d(img_channels, n_filters, kernel_size, 
                      padding=1, bias=False),
            nn.ReLU(inplace=True)]
        for _ in range(depth-2):
            layers.append(nn.Conv2d(n_filters, n_filters, kernel_size,
                                    padding=1, bias=False))
            layers.append(norm(n_filters))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(n_filters, img_channels, kernel_size,
                                padding=1, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self.H = 28; self.W = 28; self.C = 1
    def forward(self, t, x):
        return self.dncnn(x)
    
    def forward_sum(self, state_v):
        state = state_v.view(-1, self.C, self.H, self.W)
        dyn = self.forward(0, state)
        dyn_v = dyn.view(-1, self.C * self.H * self.W)
        return dyn_v.sum(dim=0)
        
    def get_jacobian(self, state):
        # B, C, H, W = state.shape
        state_v = state.view(-1, self.C * self.H * self.W)
        jacobian = autograd.functional.jacobian(self.forward_sum, state_v)
        return jacobian.permute(1,0,2)

class ODENet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # self.odefunc = ODEfunc(depth=17, img_channels=1)
        self.odefunc = ODEfunc(depth=11, img_channels=1)
        
    def forward(self, x, integration_time=None, state_idx=[1]):
        if integration_time is None:
            integration_time = torch.tensor(self.args.TimePeriod).float()
        else:
            integration_time = torch.tensor(integration_time).float()
            
        state = odeint(self.odefunc, x, integration_time, 
            rtol=self.args.rtol, atol=self.args.atol, 
            method=self.args.ode_solver, options={'step_size':self.args.step_size})
        return state[1]
        # if state_idx == 'all':
            # return state, out
        # else:
            # return [state[idx] for idx in state_idx], out
    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value



# Potential Net
class Potential(nn.Module):
    def __init__(self, in_dim=784):
        super().__init__()
        activation = nn.Softplus()
        self.net = nn.Sequential(*[
                                    nn.Linear(in_dim, 128),
                                    activation,
                                    nn.Linear(128, 256),
                                    activation,
                                    nn.Linear(256, 128),
                                    activation,
                                    nn.Linear(128, 64),
                                    activation,
                                    nn.Linear(64, 1),
                                    activation,
                                    ])
    def forward(self, state):
        out = self.net(state)
        # return out
        # shortcut = state.pow(2).sum(dim=1, keepdim=True) * 1e-1
        shortcut = state.pow(2).sum(dim=1, keepdim=True) * 1e-8
        return out + shortcut
    
class GradField(nn.Module):
    def __init__(self):
        super().__init__()
        self.potential = Potential()
        self.nfe = 0
        
    def forward(self, t, state):
        # state in (B, N)
        self.nfe += 1
        B, C, H, W = state.shape
        state_v = state.view(B,-1)
        state_v.requires_grad_()
        total_energy = self.potential(state_v).sum()
        dynamics_v = -1 * autograd.grad(total_energy, state_v, create_graph=True)[0]
        return dynamics_v.view(B,C,H,W)
    
    def forward_sum(self, state_v):
        # B, C, H, W = state.shape
        # state_v = state.view(B,-1)
        total_energy = self.potential(state_v).sum()
        dynamics_v = -1 * autograd.grad(total_energy, state_v, create_graph=True)[0]
        return dynamics_v.sum(dim=0)
        
    def get_jacobian(self, state):
        B, C, H, W = state.shape
        state_v = state.view(B,-1)
        # state.requires_grad_()
        return autograd.functional.jacobian(self.forward_sum, state_v).permute(1,0,2)
    
    
class PotentialNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.odefunc = GradField()
        # self.classifier = nn.Linear(2,4)
        
    def forward(self, x, integration_time=None, state_idx=[1]):
        if integration_time is None:
            integration_time = torch.tensor(self.args.TimePeriod).float()
        else:
            integration_time = torch.tensor(integration_time).float()
            
        state = odeint(self.odefunc, x, integration_time, 
            rtol=self.args.rtol, atol=self.args.atol, 
            method=self.args.ode_solver, options={'step_size':self.args.step_size})
        return state[1]
        # if state_idx == 'all':
            # return state, out
        # else:
            # return [state[idx] for idx in state_idx], out
    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value









if __name__ == '__main__':
    model = DnCNN(depth=6)
    input = torch.rand((1, 1, 100, 100))
    output = model(input)
    print(output.shape)
    pass