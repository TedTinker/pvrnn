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
            nn.Linear(4 * args.hidden_size, self.gen_shape[0] * self.gen_shape[1] * self.gen_shape[2]),
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
            nn.Linear(4 * args.hidden_size, args.hidden_size), 
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.hidden_size), 
            nn.PReLU(),
            nn.Linear(args.hidden_size, spe_size))
                
    def forward(self, obs, action, h):
        episodes, steps = episodes_steps(action)
        rgbd = self.rgbd_out_lin(torch.cat((obs, action, h), dim=-1)).view((episodes, steps, self.gen_shape[0], self.gen_shape[1], self.gen_shape[2]))
        rgbd_pred = rnn_cnn(self.rgbd_out, rgbd).permute(0, 1, 3, 4, 2)
        spe_pred  = self.spe_out(torch.cat((obs, action, h), dim=-1))
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
        
        

class Forward(nn.Module): 
    
    def __init__(self, args = default_args):
        super(Forward, self).__init__()
        
        self.args = default_args
        self.layers = len(args.time_scales)
                
        self.obs_in = Obs_IN(args)
        self.action_in = Action_IN(args)
        
        zp_mu_layers  = []
        zp_std_layers = []
        zq_mu_layers  = []
        zq_std_layers = []
        mtrnn_layers  = []
        
        for layer in range(self.layers): 
        
            zp_mu_layers.append(nn.Sequential(
                nn.Linear(args.hidden_size, args.hidden_size), 
                nn.PReLU(),
                nn.Linear(args.hidden_size, args.state_size),
                nn.Tanh()))
            zp_std_layers.append(nn.Sequential(
                nn.Linear(args.hidden_size, args.hidden_size), 
                nn.PReLU(),
                nn.Linear(args.hidden_size, args.state_size),
                nn.Softplus()))
            
            zq_mu_layers.append(nn.Sequential(
                nn.Linear(args.hidden_size + \
                    # rgbd, speed, action, or...
                    (args.hidden_size * 3 if layer == 0 \
                    # ...hq from lower layer
                    else args.hidden_size), args.hidden_size), 
                nn.PReLU(),
                nn.Linear(args.hidden_size, args.state_size),
                nn.Tanh()))
            zq_std_layers.append(nn.Sequential(
                nn.Linear(args.hidden_size + \
                    # rgbd, speed, action, or
                    (args.hidden_size * 3 if layer == 0 \
                    # hq from lower layer
                    else args.hidden_size), args.hidden_size), 
                nn.PReLU(),
                nn.Linear(args.hidden_size, args.state_size),
                nn.Softplus()))
            
            mtrnn_layers.append(MTRNN(
                input_size = args.state_size + (args.hidden_size if layer + 1 < self.layers else 0),
                hidden_size = args.hidden_size, 
                time_constant = args.time_scales[layer]))
            
        self.zp_mu_layers  = nn.ModuleList(zp_mu_layers)
        self.zp_std_layers = nn.ModuleList(zp_std_layers)
        self.zq_mu_layers  = nn.ModuleList(zq_mu_layers)
        self.zq_std_layers = nn.ModuleList(zq_std_layers)
        self.mtrnn_layers  = nn.ModuleList(mtrnn_layers)
        
        self.predict_obs = Obs_OUT(args)
        
        self.apply(init_weights)
        
    def p(self, hq_m1_list = None, episodes = 1):
        if(hq_m1_list == None): 
            hq_m1_list  = [torch.zeros(episodes, 1, self.args.hidden_size)] * self.layers
            zp_mu_list  = [torch.zeros(episodes, 1, self.args.state_size)]  * self.layers
            zp_std_list = [torch.ones( episodes, 1, self.args.state_size)]  * self.layers
        else:
            zp_mu_list = [] ; zp_std_list = []
            for layer in range(self.layers):
                zp_mu, zp_std = var(hq_m1_list[layer], self.zp_mu_layers[layer], self.zp_std_layers[layer], self.args)
                zp_mu_list.append(zp_mu) ; zp_std_list.append(zp_std) 
        zp_list = [] ; hp_list = []
        for layer in range(self.layers):
            zp_list.append(sample(zp_mu_list[layer], zp_std_list[layer]))
            if(layer+1 == self.layers): hp, _ = self.mtrnn_layers[layer](zp_list[layer], hq_m1_list[layer])
            else:                       hp, _ = self.mtrnn_layers[layer](torch.cat([zp_list[layer], hq_m1_list[layer+1]], dim = -1), hq_m1_list[layer]) 
            hp_list.append(hp)
        return(zp_mu_list, zp_std_list, hp_list)
    
    def q(self, rgbd, speed, action, hq_m1_list = None):
        if(len(rgbd.shape)   == 4): rgbd   = rgbd.unsqueeze(1)
        if(len(speed.shape)  == 2): speed  = speed.unsqueeze(1)
        if(len(action.shape) == 2): action = action.unsqueeze(1)
        episodes, steps = episodes_steps(rgbd)
        if(hq_m1_list == None):     hq_m1_list = [torch.zeros(episodes, steps, self.args.hidden_size)] * self.layers
        obs = self.obs_in(rgbd, speed)
        action = self.action_in(action)
        zq_mu_list = [] ; zq_std_list = [] ; zq_list = [] ; hq_list = []
        for layer in range(self.layers):
            if(layer == 0): zq_mu, zq_std = var(torch.cat((hq_m1_list[layer], obs, action),      dim=-1), self.zq_mu_layers[layer], self.zq_std_layers[layer], self.args)        
            else:           zq_mu, zq_std = var(torch.cat((hq_m1_list[layer], hq_list[layer-1]), dim=-1), self.zq_mu_layers[layer], self.zq_std_layers[layer], self.args)  
            zq_mu_list.append(zq_mu) ; zq_std_list.append(zq_std) ; zq_list.append(sample(zq_mu, zq_std))
            if(layer+1 == self.layers): hq, _ = self.mtrnn_layers[layer](zq_list[layer], hq_m1_list[layer])
            else:                       hq, _ = self.mtrnn_layers[layer](torch.cat([zq_list[layer], hq_m1_list[layer+1]], dim = -1), hq_m1_list[layer]) 
            hq_list.append(hq)
        return(zq_mu_list, zq_std_list, hq_list)
        
    def predict(self, rgbd, speed, action, h): 
        if(len(rgbd.shape)   == 4): rgbd   = rgbd.unsqueeze(1)
        if(len(speed.shape)  == 2): speed  = speed.unsqueeze(1)
        if(len(action.shape) == 2): action = action.unsqueeze(1)
        if(len(h[0].shape)   == 2): h[0]   = h[0].unsqueeze(1)
        obs = self.obs_in(rgbd, speed)
        action = self.action_in(action)
        pred_rgbd, pred_speed = self.predict_obs(obs, action, h[0])
        return(pred_rgbd, pred_speed)
    
    def forward(self, rgbd, speed, action):
        episodes, steps = episodes_steps(rgbd)
        zp_mu_lists = [] ; zp_std_lists = [] ; zp_rgbd_pred_list = [] ; zp_speed_pred_list = [] ; hp_lists = []
        zq_mu_lists = [] ; zq_std_lists = [] ; zq_rgbd_pred_list = [] ; zq_speed_pred_list = [] ; hq_lists = [None]
        for step in range(steps):
            zp_mu_list, zp_std_list, hp_list = self.p(hq_lists[step], episodes = episodes)
            zq_mu_list, zq_std_list, hq_list = self.q(rgbd[:,step], speed[:,step], action[:,step], hq_lists[step])
            zp_rgbd_pred, zp_speed_pred = self.predict(rgbd[:,step], speed[:,step], action[:,step], hp_list)
            zq_rgbd_pred, zq_speed_pred = self.predict(rgbd[:,step], speed[:,step], action[:,step], hq_list)
            
            zp_mu_lists.append(zp_mu_list) ; zp_std_lists.append(zp_std_list) ; hp_lists.append(hp_list)
            zp_rgbd_pred_list.append(zp_rgbd_pred) ; zp_speed_pred_list.append(zp_speed_pred)
            zq_mu_lists.append(zq_mu_list) ; zq_std_lists.append(zq_std_list) ; hq_lists.append(hq_list)
            zq_rgbd_pred_list.append(zq_rgbd_pred) ; zq_speed_pred_list.append(zq_speed_pred)
            
        hq_lists[0] = [torch.zeros(episodes, 1, self.args.hidden_size)] * self.layers
        return(
            (zp_mu_lists, zp_std_lists, zp_rgbd_pred_list, zp_speed_pred_list, hp_lists), 
            (zq_mu_lists, zq_std_lists, zq_rgbd_pred_list, zq_speed_pred_list, hq_lists))



class Actor(nn.Module):

    def __init__(self, args = default_args):
        super(Actor, self).__init__()
        
        self.args = args
        
        self.obs_in = Obs_IN(args)
        
        self.lin = nn.Sequential(
            nn.PReLU(),
            nn.Linear(3 * args.hidden_size, args.hidden_size),
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

    def forward(self, rgbd, speed, h = None):
        episodes, steps = episodes_steps(rgbd)
        if(h == None): h = torch.zeros(episodes, steps, self.args.hidden_size)
        else: h = h[0]
        obs = self.obs_in(rgbd, speed)
        x = self.lin(torch.cat([obs, h], dim = -1))
        mu, std = var(x, self.mu, self.std, self.args)
        x = sample(mu, std)
        action = torch.tanh(x)
        log_prob = Normal(mu, std).log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = torch.mean(log_prob, -1).unsqueeze(-1)
        return(action, log_prob, None)
    
    
    
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

    def forward(self, rgbd, speed, action, h = None):
        obs = self.obs_in(rgbd, speed)
        action = self.action_in(action)
        h, _ = self.gru(torch.cat((obs, action), dim=-1), h)
        Q = self.lin(self.h_in(h))
        return(Q, h)
    


if __name__ == "__main__":
    
    args = default_args
    e = 3 ; s = 3
    
    
    forward = Forward(args)
    
    print("\n\n")
    print(forward)
    print()
    print(torch_summary(forward, ((e, s, args.image_size, args.image_size, 4), (e, s, spe_size), (e, s, action_size))))
    
    
    
    args.time_scales = [1, .85]
    forward = Forward(args)
    
    print("\n\n")
    print(forward)
    print()
    print(torch_summary(forward, ((e, s, args.image_size, args.image_size, 4), (e, s, spe_size), (e, s, action_size))))
    
    
    
    actor = Actor(args)
    
    print("\n\n")
    print(actor)
    print()
    print(torch_summary(actor, ((e, s, args.image_size, args.image_size, 4), (e, s, spe_size))))
    
    
    
    critic = Critic(args)
    
    print("\n\n")
    print(critic)
    print()
    print(torch_summary(critic, ((e, s, args.image_size, args.image_size, 4), (e, s, spe_size), (e, s, action_size))))

# %%