#%% 

from math import log2
import torch
from torch import nn 
import torch.nn.functional as F
from torch.distributions import Normal
from torchinfo import summary as torch_summary
from torchgan.layers import SelfAttention2d

from utils import default_args, init_weights, ConstrainedConv2d, ConstrainedConvTranspose2d, Ted_Conv2d, print
spe_size = 1 ; action_size = 2 ; obs_num = 3



def episodes_steps(this):
    return(this.shape[0], this.shape[1])

def var(x, mu_func, std_func, args):
    mu = mu_func(x)
    std = torch.clamp(std_func(x), min = args.std_min, max = args.std_max)
    return(mu, std)

def sample(mu, std):
    e = Normal(0, 1).sample(std.shape).to("cuda" if std.is_cuda else "cpu")
    return(mu + e * std)

def rnn_cnn(do_this, to_this):
    episodes = to_this.shape[0] ; steps = to_this.shape[1]
    this = to_this.view((episodes * steps, to_this.shape[2], to_this.shape[3], to_this.shape[4]))
    this = do_this(this)
    this = this.view((episodes, steps, this.shape[1], this.shape[2], this.shape[3]))
    return(this)



class Obs_IN(nn.Module):

    def __init__(self, args = default_args):
        super(Obs_IN, self).__init__()  
        
        self.args = args
        
        rgbd_size = (1, 4, args.image_size, args.image_size)
        example = torch.zeros(rgbd_size)
        
        n_blocks = int(log2(args.image_size) - 2)
        modules = []
        modules.extend([
            ConstrainedConv2d(
                in_channels=4,
                out_channels=16,
                kernel_size=(3, 3),
                padding=(1, 1),
                padding_mode="reflect"),
            nn.PReLU(),
            nn.AvgPool2d(
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1))])
        for i in range(n_blocks):
            modules.extend([
                ConstrainedConv2d(
                    in_channels=16,
                    out_channels=16,
                    kernel_size=(3, 3),
                    padding=(1, 1),
                    padding_mode="reflect"),
                nn.PReLU(),
                nn.AvgPool2d(
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1))])
        self.rgbd_in = nn.Sequential(*modules)
        
        example = self.rgbd_in(example)
        rgbd_latent_size = example.flatten(1).shape[1]
        
        self.rgbd_in_lin = nn.Sequential(
            nn.Linear(rgbd_latent_size, args.hidden_size),
            nn.PReLU())
        
        self.speed_in = nn.Sequential(
            nn.Linear(1, args.hidden_size),
            nn.PReLU())
        
    def forward(self, rgbd, speed):
        if(len(rgbd.shape) == 4):   rgbd =  rgbd.unsqueeze(1)
        if(len(speed.shape) == 2):  speed = speed.unsqueeze(1)
        rgbd = (rgbd.permute(0, 1, 4, 2, 3) * 2) - 1
        rgbd = rnn_cnn(self.rgbd_in, rgbd).flatten(2)
        rgbd = self.rgbd_in_lin(rgbd)
        speed = (speed - self.args.min_speed) / (self.args.max_speed - self.args.min_speed)
        speed = self.speed_in(speed)
        return(torch.cat([rgbd, speed], dim = -1))
    
    
    
class Obs_OUT(nn.Module):

    def __init__(self, args = default_args):
        super(Obs_OUT, self).__init__()  
                
        self.gen_shape = (4, 2, 2) 
        self.rgbd_out_lin = nn.Sequential(
            nn.Linear(2 * args.hidden_size, self.gen_shape[0] * self.gen_shape[1] * self.gen_shape[2]),
            nn.PReLU())
                
        n_blocks = int(log2(args.image_size))
        modules = []
        for i in range(n_blocks):
            modules.extend([
            ConstrainedConv2d(
                in_channels = self.gen_shape[0] if i == 0 else 16,
                out_channels = 16,
                kernel_size = (3,3),
                padding = (1,1),
                padding_mode = "reflect"),
            nn.PReLU()
            ])
            if i != n_blocks - 1:
                modules.extend([
                    nn.Upsample(scale_factor = 2, mode = "bilinear", align_corners = True),
                ])
        modules.extend([
            ConstrainedConv2d(
                in_channels = 16,
                out_channels = 4,
                kernel_size = (1,1))
        ])
        self.rgbd_out = nn.Sequential(*modules)
        
        self.spe_out = nn.Sequential(
            nn.Linear(2 * args.hidden_size, args.hidden_size), 
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.hidden_size), 
            nn.PReLU(),
            nn.Linear(args.hidden_size, spe_size))
                
    def forward(self, action, h):
        episodes, steps = episodes_steps(action)
        rgbd = self.rgbd_out_lin(torch.cat((h, action), dim=-1)).view((episodes, steps, self.gen_shape[0], self.gen_shape[1], self.gen_shape[2]))
        rgbd_pred = rnn_cnn(self.rgbd_out, rgbd).permute(0, 1, 3, 4, 2)
        spe_pred  = self.spe_out(torch.cat((h, action), dim=-1))
        return(rgbd_pred, spe_pred)
    
    
    
class Action_IN(nn.Module):
    
    def __init__(self, args = default_args):
        super(Action_IN, self).__init__()
        
        self.action_in = nn.Sequential(
            nn.Linear(action_size, args.hidden_size),
            nn.PReLU())
        
    def forward(self, action):
        if(len(action.shape) == 2):   action = action.unsqueeze(1)
        action = self.action_in(action)
        return(action)
        
        

class MTRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, time_constant):
        super(MTRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.time_constant = time_constant
        
        self.weight_ih = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))

    def forward(self, input, hx):
        linear_i = torch.mm(input, self.weight_ih.t())
        linear_h = torch.mm(hx, self.weight_hh.t())
        new_h = (1 - 1 / self.time_constant) * hx + \
                (1 / self.time_constant) * (linear_i + linear_h + self.bias).tanh()
        return new_h

class MTRNN(nn.Module):
    def __init__(self, input_size, hidden_size, time_constant):
        super(MTRNN, self).__init__()
        self.mtrnn_cell = MTRNNCell(input_size, hidden_size, time_constant)

    def forward(self, input, hx=None):
        episodes, steps = episodes_steps(input)
        outputs = []
        for step in range(steps):  
            hx = self.mtrnn_cell(input[:, step], hx[:, step])
            outputs.append(hx)
        outputs = torch.stack(outputs, dim = 1)
        return outputs[:, -1].unsqueeze(1), outputs
        
        

class Forward(nn.Module): # PVRNN!
    
    def __init__(self, args = default_args):
        super(Forward, self).__init__()
        
        self.args = default_args
                
        self.obs_in = Obs_IN(args)
        self.action_in = Action_IN(args)
        
        self.layers = len(args.time_scales)
        zp_mu_layers  = []
        zp_std_layers = []
        zq_mu_layers  = []
        zq_std_layers = []
        mtrnn_layers  = []
        
        for layer in range(self.layers):
        
            zp_mu_layers.append(nn.Sequential(
                nn.Linear(args.hidden_size + (args.hidden_size if layer == 0 else args.state_size), args.hidden_size), 
                nn.PReLU(),
                nn.Linear(args.hidden_size, args.state_size),
                nn.Tanh()))
            zp_std_layers.append(nn.Sequential(
                nn.Linear(args.hidden_size + (args.hidden_size if layer == 0 else args.state_size), args.hidden_size), 
                nn.PReLU(),
                nn.Linear(args.hidden_size, args.state_size),
                nn.Softplus()))
            
            zq_mu_layers.append(nn.Sequential(
                nn.Linear(args.hidden_size + (args.hidden_size * obs_num if layer == 0 else args.state_size), args.hidden_size), 
                nn.PReLU(),
                nn.Linear(args.hidden_size, args.state_size),
                nn.Tanh()))
            zq_std_layers.append(nn.Sequential(
                nn.Linear(args.hidden_size + (args.hidden_size * obs_num if layer == 0 else args.state_size), args.hidden_size), 
                nn.PReLU(),
                nn.Linear(args.hidden_size, args.state_size),
                nn.Softplus()))
            
            mtrnn_layers.append(MTRNN(
                input_size = args.state_size + (args.hidden_size if layer+1 != self.layers else 0),
                hidden_size = args.hidden_size, 
                time_constant = args.time_scales[layer]))
            
        self.zp_mu_layers  = nn.ModuleList(zp_mu_layers)
        self.zp_std_layers = nn.ModuleList(zp_std_layers)
        self.zq_mu_layers  = nn.ModuleList(zq_mu_layers)
        self.zq_std_layers = nn.ModuleList(zq_std_layers)
        self.mtrnn_layers  = nn.ModuleList(mtrnn_layers)
        
        self.predict_obs = Obs_OUT(args)
        
        self.apply(init_weights)
        
    def forward(self, rgbd, spe, prev_action, hq_m1_list = None):
        episodes, steps = episodes_steps(rgbd)
        if(hq_m1_list == None): hq_m1_list = [torch.zeros(episodes, 1, self.args.hidden_size)] * self.layers
        
        # Information goes up the layers.
        obs = self.obs_in(rgbd, spe)
        prev_action = self.action_in(prev_action)
        zp_mus = [] ; zp_stds = [] ; zps = []
        zq_mus = [] ; zq_stds = [] ; zqs = []
        for layer in range(self.layers):
            relu_hq_m1 = F.relu(hq_m1_list[layer])
            if(layer == 0):
                zp_mu, zp_std = var(torch.cat((relu_hq_m1,      prev_action), dim=-1), self.zp_mu_layers[layer], self.zp_std_layers[layer], self.args)
                zq_mu, zq_std = var(torch.cat((relu_hq_m1, obs, prev_action), dim=-1), self.zq_mu_layers[layer], self.zq_std_layers[layer], self.args)        
            else:
                zp_mu, zp_std = var(torch.cat((relu_hq_m1, zps[-1]),          dim=-1), self.zp_mu_layers[layer], self.zp_std_layers[layer], self.args)
                zq_mu, zq_std = var(torch.cat((relu_hq_m1, zqs[-1]),          dim=-1), self.zq_mu_layers[layer], self.zq_std_layers[layer], self.args)        
            zp_mus.append(zp_mu) ; zp_stds.append(zp_std)
            zq_mus.append(zq_mu) ; zq_stds.append(zq_std)
            zp = sample(zp_mu, zp_std) ; zps.append(zp)
            zq = sample(zq_mu, zq_std) ; zqs.append(zq)
        
        # Then understanding goes down the layers.
        hq_list = [None] * self.layers
        for layer in range(self.layers - 1, -1, -1):
            if(layer == self.layers - 1):
                hq, _ = self.mtrnn_layers[layer](zqs[layer],                                          hq_m1_list[layer])
            else:
                hq, _ = self.mtrnn_layers[layer](torch.cat((zqs[layer], hq_list[layer+1]), dim = -1), hq_m1_list[layer])
            hq_list[layer] = hq
            
        return((zp_mus, zp_stds), (zq_mus, zq_stds), hq_list)
        
    def predict(self, action, z_mus, z_stds, h_m1_list, quantity = 1):
        if(len(action.shape) == 2):        action =  action.unsqueeze(1)
        if(len(h_m1_list[0].shape) == 2):  h_m1_list[0] = h_m1_list[0].unsqueeze(1)
        z_mu = z_mus[0] ; z_std = z_stds[0]
        h_m1 = h_m1_list[0] ; h_m1_up = h_m1_list[1] if self.layers > 1 else None
        h, _ = self.mtrnn_layers[0](z_mu if h_m1_up == None else torch.cat([z_mu, h_m1_up], dim = -1), h_m1)       
        action = self.action_in(action)
        mu_pred = self.predict_obs(action, h)
                
        std_preds = []
        for _ in range(quantity):
            z = sample(z_mu, z_std)
            h, _ = self.mtrnn_layers[0](z if h_m1_up == None else torch.cat([z, h_m1_up], dim = -1), h_m1)   
            std_preds.append(self.predict_obs(action, h))
        return(mu_pred, std_preds)
    
    def forward_complete(self, obs, actions, quantity = 1):
        episodes, steps = episodes_steps(obs)
        hq_m1_list = [torch.zeros(episodes, 1, self.args.hidden_size)] * self.layers
        zq_pred_list = [] ; zp_pred_list = [] ; hq_lists = []
        for step in steps:
            (zp_mus, zp_stds), (zq_mus, zq_stds), hq_list = self(obs[:, step], actions[:, step-1], hq_m1_list)
            zq_pred_list.append(self.predict(actions[:, step], zq_mus, zq_stds, hq_m1_list, quantity))
            zp_pred_list.append(self.predict(actions[:, step], zp_mus, zp_stds, hq_m1_list, quantity))
            hq_lists.append(hq_list)
            hq_m1_list = hq_list
        return(zq_pred_list, zp_pred_list, hq_lists)



class Actor(nn.Module):

    def __init__(self, args = default_args):
        super(Actor, self).__init__()
        
        self.args = args
        
        self.obs_in = Obs_IN(args)
        self.action_in = Action_IN(args)
        
        self.h_in = nn.Sequential(
            nn.PReLU())
        
        self.gru = nn.GRU(
            input_size =  3 * args.hidden_size,
            hidden_size = args.hidden_size,
            batch_first = True)
        
        self.mu = nn.Sequential(
            nn.Linear(args.hidden_size, action_size))
        self.std = nn.Sequential(
            nn.Linear(args.hidden_size, action_size),
            nn.Softplus())

        self.apply(init_weights)
        self.to(args.device)

    def forward(self, rgbd, spe, prev_action, h = None):
        if(len(rgbd.shape) == 4): rgbd = rgbd.unsqueeze(1)
        if(len(spe.shape) == 2):  spe =  spe.unsqueeze(1)
        obs = self.obs_in(rgbd, spe)
        prev_action = self.action_in(prev_action)
        h, _ = self.gru(torch.cat((obs, prev_action), dim=-1), h)
        relu_h = self.h_in(h)
        mu, std = var(relu_h, self.mu, self.std, self.args)
        x = sample(mu, std)
        action = torch.tanh(x)
        log_prob = Normal(mu, std).log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = torch.mean(log_prob, -1).unsqueeze(-1)
        return(action, log_prob, h)
    
    
    
class Critic(nn.Module):

    def __init__(self, args = default_args):
        super(Critic, self).__init__()
        
        self.args = args
        
        self.obs_in = Obs_IN(args)
        self.action_in = Action_IN(args)
        
        self.h_in = nn.Sequential(
            nn.PReLU())
        
        self.gru = nn.GRU(
            input_size =  3 * args.hidden_size,
            hidden_size = args.hidden_size,
            batch_first = True)
        
        self.lin = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.PReLU(),
            nn.Linear(args.hidden_size, 1))

        self.apply(init_weights)
        self.to(args.device)

    def forward(self, rgbd, spe, action, h = None):
        if(len(rgbd.shape) == 4): rgbd = rgbd.unsqueeze(1)
        if(len(spe.shape) == 2):  spe =  spe.unsqueeze(1)
        obs = self.obs_in(rgbd, spe)
        action = self.action_in(action)
        h, _ = self.gru(torch.cat((obs, action), dim=-1), h)
        Q = self.lin(self.h_in(h))
        return(Q, h)
    
    
    
class Actor_HQ(nn.Module):

    def __init__(self, args = default_args):
        super(Actor_HQ, self).__init__()
        
        self.args = args
        
        self.lin = nn.Sequential(
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.PReLU())
        self.mu = nn.Sequential(
            nn.Linear(args.hidden_size, action_size))
        self.std = nn.Sequential(
            nn.Linear(args.hidden_size, action_size),
            nn.Softplus())

        self.apply(init_weights)
        self.to(args.device)

    def forward(self, h):
        x = self.lin(h[0])
        mu, std = var(x, self.mu, self.std, self.args)
        x = sample(mu, std)
        action = torch.tanh(x)
        log_prob = Normal(mu, std).log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = torch.mean(log_prob, -1).unsqueeze(-1)
        return(action, log_prob, None)
    
    
    
class Critic_HQ(nn.Module):

    def __init__(self, args = default_args):
        super(Critic_HQ, self).__init__()
        
        self.args = args
        
        self.lin = nn.Sequential(
            nn.PReLU(),
            nn.Linear(args.hidden_size + action_size, args.hidden_size),
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.PReLU(),
            nn.Linear(args.hidden_size, 1))

        self.apply(init_weights)
        self.to(args.device)

    def forward(self, h, action):
        Q = self.lin(torch.cat((h, action), dim=-1))
        return(Q, None)
    


if __name__ == "__main__":
    
    args = default_args
    args.dkl_rate = 1
    args.image_size = 8
    
    
    
    forward = Forward(args)
    
    print("\n\n")
    print(forward)
    print()
    print(torch_summary(forward, ((3, 1, args.image_size, args.image_size, 4), (3, 1, spe_size), (3, 1, action_size))))
    
    

    actor = Actor(args)
    
    print("\n\n")
    print(actor)
    print()
    print(torch_summary(actor, ((3, 1, args.image_size, args.image_size, 4), (3, 1, spe_size), (3, 1, action_size))))
    
    
    
    critic = Critic(args)
    
    print("\n\n")
    print(critic)
    print()
    print(torch_summary(critic, ((3, 1, args.image_size, args.image_size, 4), (3, 1, spe_size), (3, 1, action_size))))
    
    
    
    actor = Actor_HQ(args)
    
    print("\n\n")
    print(actor)
    print()
    print(torch_summary(actor, ((3, 1, args.hidden_size))))
    
    
    
    critic = Critic_HQ(args)
    
    print("\n\n")
    print(critic)
    print()
    print(torch_summary(critic, ((3, 1, args.hidden_size), (3, 1, action_size))))

# %%
