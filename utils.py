#%% 

import builtins

def print(*args, **kwargs):
    kwargs["flush"] = True
    builtins.print(*args, **kwargs)



import matplotlib
import matplotlib.pyplot as plt

font = {'family' : 'sans-serif',
        #'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

#plt.rcParams['axes.labelweight'] = 'bold'
#plt.rcParams['axes.titleweight'] = 'bold'



import datetime 

start_time = datetime.datetime.now()
    
def duration(start_time = start_time):
    change_time = datetime.datetime.now() - start_time
    change_time = change_time - datetime.timedelta(microseconds=change_time.microseconds)
    return(change_time)

def estimate_total_duration(proportion_completed, start_time=start_time):
    if(proportion_completed != 0): 
        so_far = datetime.datetime.now() - start_time
        estimated_total = so_far / proportion_completed
        estimated_total = estimated_total - datetime.timedelta(microseconds=estimated_total.microseconds)
    else: estimated_total = "?:??:??"
    return(estimated_total)



import argparse, ast, os, pickle
from math import exp, pi
import numpy as np
from time import sleep

if(os.getcwd().split("/")[-1] != "pvrnn"): os.chdir("pvrnn")

import torch
from torch import nn 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

def literal(arg_string): return(ast.literal_eval(arg_string))

# Meta 
parser.add_argument("--arg_title",          type=str,        default = "default") 
parser.add_argument("--arg_name",           type=str,        default = "default") 
parser.add_argument("--agents",             type=int,        default = 36)
parser.add_argument("--previous_agents",    type=int,        default = 0)
parser.add_argument("--init_seed",          type=float,      default = 777)
parser.add_argument('--device',             type=str,        default = device)
parser.add_argument('--comp',               type=str,        default = "deigo")

# Maze 
parser.add_argument('--hard_maze',          type=literal,    default = False)
parser.add_argument('--maze_list',          type=literal,    default = ["t"])
parser.add_argument('--max_steps',          type=int,        default = 10)
parser.add_argument('--step_lim_punishment',type=float,      default = -1)
parser.add_argument('--wall_punishment',    type=float,      default = -1)
parser.add_argument('--non_one',            type=float,      default = -1)
parser.add_argument('--default_reward',     type=literal,    default = [(1, 1)])  # ((weight, reward), (weight, reward))
parser.add_argument('--better_reward',      type=literal,    default = [(1, 0), (1, 10)])
parser.add_argument('--randomness',         type=float,      default = 0)
parser.add_argument('--random_steps',       type=int,        default = 1)
parser.add_argument('--random_by_choice',   type=bool,       default = False)
parser.add_argument('--step_cost',          type=float,      default = .99)

# Hard Maze
parser.add_argument('--body_size',          type=float,      default = 2)    
parser.add_argument('--image_size',         type=int,        default = 8)
parser.add_argument('--max_yaw_change',     type=float,      default = pi/2)
parser.add_argument('--min_speed',          type=float,      default = 75)
parser.add_argument('--max_speed',          type=float,      default = 150)
parser.add_argument('--steps_per_step',     type=int,        default = 5)
parser.add_argument('--speed_scalar',       type=float,      default = .0001)
parser.add_argument('--boxes_high',         type=int,        default = 1)

# Module 
parser.add_argument('--hidden_size',        type=int,        default = 32)   
parser.add_argument('--state_size',         type=int,        default = 32)
parser.add_argument('--time_scales',        type=literal,    default = [1])
parser.add_argument('--actor_hq',           type=literal,    default = True)
parser.add_argument('--critic_hq',          type=literal,    default = False)
parser.add_argument('--forward_lr',         type=float,      default = .01)
parser.add_argument('--alpha_lr',           type=float,      default = .01) 
parser.add_argument('--actor_lr',           type=float,      default = .01)
parser.add_argument('--critic_lr',          type=float,      default = .01)
parser.add_argument('--action_prior',       type=str,        default = "normal")
parser.add_argument("--tau",                type=float,      default = .1)      # For soft-updating target critics

# Complexity 
parser.add_argument('--std_min',            type=int,        default = exp(-20))
parser.add_argument('--std_max',            type=int,        default = exp(2))
parser.add_argument("--beta",               type=literal,    default = [0])

# Entropy
parser.add_argument("--alpha",              type=str,        default = 0)        # Soft-Actor-Critic entropy aim
parser.add_argument("--target_entropy",     type=float,      default = -2)       # Soft-Actor-Critic entropy aim

# Curiosity
parser.add_argument("--curiosity",          type=str,        default = "none")     # Which kind of curiosity
parser.add_argument("--dkl_max",            type=float,      default = 1)        
parser.add_argument("--naive_eta",          type=float,      default = 10)        # Scale curiosity
parser.add_argument("--target_naive_curiosity",type=float,   default = -2)        # Curiosity aim
parser.add_argument("--free_eta",           type=literal,    default = [10])      # Scale curiosity
parser.add_argument("--target_free_curiosity", type=literal, default = [-2])      # Curiosity aims

# Memory buffer
parser.add_argument('--capacity',           type=int,        default = 250)

# Training
parser.add_argument('--epochs',             type=literal,    default = [1000])
parser.add_argument('--steps_per_epoch',    type=int,        default = 10)
parser.add_argument('--batch_size',         type=int,        default = 128)
parser.add_argument('--elbo_num',           type=int,        default = 1)
parser.add_argument('--GAMMA',              type=float,      default = .9)
parser.add_argument("--d",                  type=int,        default = 2)        # Delay to train actors
parser.add_argument('--retroactive_reward', type=literal,    default = False)
parser.add_argument('--retro_step_cost',    type=float,      default = .99)

# Saving data
parser.add_argument('--keep_data',           type=int,        default = 1)

parser.add_argument('--epochs_per_pred_list',type=int,        default = 10000000)
parser.add_argument('--agents_per_pred_list',type=int,        default = 1)
parser.add_argument('--episodes_in_pred_list',type=int,       default = 1)
parser.add_argument('--samples_per_pred',    type=int,        default = 1)

parser.add_argument('--epochs_per_pos_list', type=int,        default = 100)
parser.add_argument('--agents_per_pos_list', type=int,        default = -1)
parser.add_argument('--episodes_in_pos_list',type=int,        default = 1)

parser.add_argument('--epochs_per_agent_list',type=int,       default = 100000)#100)
parser.add_argument('--agents_per_agent_list',type=int,       default = 1)



try:
    default_args = parser.parse_args([])
    try:    args    = parser.parse_args()
    except: args, _ = parser.parse_known_args()
except:
    import sys ; sys.argv=[''] ; del sys           # Comment this out when using bash
    default_args = parser.parse_args([])
    try:    args    = parser.parse_args()
    except: args, _ = parser.parse_known_args()

for arg in vars(default_args):
    if(getattr(default_args, arg) == "None"):  default_args.arg = None
    if(getattr(default_args, arg) == "True"):  default_args.arg = True
    if(getattr(default_args, arg) == "False"): default_args.arg = False
    if(getattr(args, arg) == "None"):  args.arg = None
    if(getattr(args, arg) == "True"):  args.arg = True
    if(getattr(args, arg) == "False"): args.arg = False
    
default_args.steps_per_epoch = default_args.max_steps
args.steps_per_epoch = args.max_steps

def extend_list_to_match_length(target_list, length, value):
    while len(target_list) < length:
        target_list.append(value)
    return target_list

max_length = max(len(args.time_scales), len(args.beta), len(args.free_eta))
args.time_scales = extend_list_to_match_length(args.time_scales, max_length, 1)
args.beta = extend_list_to_match_length(args.beta, max_length, 0)
args.free_eta = extend_list_to_match_length(args.free_eta, max_length, 0)




args_not_in_title = ["arg_title", "id", "agents", "previous_agents", "init_seed", "hard_maze", "maze_list", "keep_data", "epochs_per_pred_list", "episodes_in_pred_list", "agents_per_pred_list", "epochs_per_pos_list", "episodes_in_pos_list", "agents_per_pos_list"]
def get_args_title(default_args, args):
    if(args.arg_title[:3] == "___"): return(args.arg_title)
    name = "" ; first = True
    arg_list = list(vars(default_args).keys())
    arg_list.insert(0, arg_list.pop(arg_list.index("arg_name")))
    for arg in arg_list:
        if(arg in args_not_in_title): pass 
        else: 
            default, this_time = getattr(default_args, arg), getattr(args, arg)
            if(this_time == default): pass
            elif(arg == "arg_name"):
                name += "{} (".format(this_time)
            else: 
                if first: first = False
                else: name += ", "
                name += "{}: {}".format(arg, this_time)
    if(name == ""): name = "default" 
    else:           name += ")"
    if(name.endswith(" ()")): name = name[:-3]
    parts = name.split(',')
    name = "" ; line = ""
    for i, part in enumerate(parts):
        if(len(line) > 50 and len(part) > 2): name += line + "\n" ; line = ""
        line += part
        if(i+1 != len(parts)): line += ","
    name += line
    return(name)

args.arg_title = get_args_title(default_args, args)

try: os.mkdir("saved")
except: pass
folder = "saved/" + args.arg_name
if(args.arg_title[:3] != "___" and not args.arg_name in ["default", "finishing_dictionaries", "plotting", "plotting_predictions", "plotting_positions"]):
    try: os.mkdir(folder)
    except: pass
    try: os.mkdir("saved/thesis_pics")
    except: pass
    try: os.mkdir("saved/thesis_pics/final")
    except: pass
if(default_args.alpha == "None"): default_args.alpha = None
if(args.alpha == "None"):         args.alpha = None

if(args == default_args): print("Using default arguments.")
else:
    for arg in vars(default_args):
        default, this_time = getattr(default_args, arg), getattr(args, arg)
        if(this_time == default): pass
        else: print("{}:\n\tDefault:\t{}\n\tThis time:\t{}".format(arg, default, this_time))



def init_weights(m):
    try:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
    except: pass
    
class ConstrainedConv2d(nn.Conv2d):
    def forward(self, input):
        return nn.functional.conv2d(input, self.weight.clamp(min=-1.0, max=1.0), self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)
        
class ConstrainedConvTranspose2d(nn.ConvTranspose2d):
    def forward(self, input):
        return nn.functional.conv_transpose2d(input, self.weight.clamp(min=-1.0, max=1.0), self.bias, self.stride,
                                    self.padding, self.output_padding, self.groups, self.dilation)
        
class Ted_Conv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernels = [(1,1),(3,3),(5,5)]):
        super(Ted_Conv2d, self).__init__()
        
        self.Conv2ds = nn.ModuleList()
        for kernel, out_channel in zip(kernels, out_channels):
            padding = ((kernel[0]-1)//2, (kernel[1]-1)//2)
            layer = nn.Sequential(
                ConstrainedConv2d(
                    in_channels = in_channels,
                    out_channels = out_channel,
                    kernel_size = kernel,
                    padding = padding,
                    padding_mode = "reflect"),
                nn.PReLU())
            self.Conv2ds.append(layer)
                
    def forward(self, x):
        y = []
        for Conv2d in self.Conv2ds: y.append(Conv2d(x)) 
        return(torch.cat(y, dim = -3))


    
def dkl(mu_1, std_1, mu_2, std_2):
    std_1 = std_1**2
    std_2 = std_2**2
    term_1 = (mu_2 - mu_1)**2 / std_2 
    term_2 = std_1 / std_2 
    term_3 = torch.log(term_2)
    out = (.5 * (term_1 + term_2 - term_3 - 1))
    out = torch.nan_to_num(out)
    return(out)



real_names = {
    "d"  : "No Entropy, No Curiosity",
    "e"  : "Entropy",
    "n"  : "Prediction Error Curiosity",
    "en" : "Entropy and Prediction Error Curiosity",
    "f"  : "Hidden State Curiosity",
    "ef" : "Entropy and Hidden State Curiosity"
}

def add_this(name):
    keys, values = [], []
    for key, value in real_names.items(): keys.append(key) ; values.append(value)
    for key, value in zip(keys, values):  
        new_key = key + "_" + name 
        real_names[new_key] = value
add_this("hard")
add_this("many")

short_real_names = {
    "d"  : "N",
    "e"  : "E",
    "n"  : "P",
    "en" : "EP",
    "f"  : "H",
    "ef" : "EH"
}

def add_this(name):
    keys, values = [], []
    for key, value in short_real_names.items(): keys.append(key) ; values.append(value)
    for key, value in zip(keys, values):  
        new_key = key + "_" + name 
        short_real_names[new_key] = value
add_this("hard")
add_this("many")

maze_real_names = {
    "t" : "Biased T-Maze",
    "alt" : "Biased T-Maze",
    "1" : "T-Maze",
    "2" : "Double T-Maze",
    "3" : "Triple T-Maze",
}



def load_dicts(args):
    if(os.getcwd().split("/")[-1] != "saved"): os.chdir("saved")
    plot_dicts = [] ; min_max_dicts = []
        
    complete_order = args.arg_title[3:-3].split("+")
    order = [o for o in complete_order if not o in ["empty_space", "break"]]

    for name in order:
        got_plot_dicts = False ; got_min_max_dicts = False
        while(not got_plot_dicts):
            try:
                with open(name + "/" + "plot_dict.pickle", "rb") as handle: 
                    plot_dicts.append(pickle.load(handle)) ; got_plot_dicts = True
            except: print("Stuck trying to get {}'s plot_dicts...".format(name)) ; sleep(1)
        while(not got_min_max_dicts):
            try:
                with open(name + "/" + "min_max_dict.pickle", "rb") as handle: 
                    min_max_dicts.append(pickle.load(handle)) ; got_min_max_dicts = True 
            except: print("Stuck trying to get {}'s min_max_dicts...".format(name)) ; sleep(1)
    
    min_max_dict = {}
    for key in plot_dicts[0].keys():
        if(not key in ["args", "arg_title", "arg_name", "pred_lists", "pos_lists", "agents_lists", "spot_names", "steps"]):
            minimum = None ; maximum = None
            for mm_dict in min_max_dicts:
                if(mm_dict[key] != (None, None)):
                    if(minimum == None):             minimum = mm_dict[key][0]
                    elif(minimum > mm_dict[key][0]): minimum = mm_dict[key][0]
                    if(maximum == None):             maximum = mm_dict[key][1]
                    elif(maximum < mm_dict[key][1]): maximum = mm_dict[key][1]
            min_max_dict[key] = (minimum, maximum)
            
    complete_easy_order = [] ; easy_plot_dicts = []
    complete_hard_order = [] ; hard_plot_dicts = []

    easy = False 
    hard = False 
    for arg_name in complete_order: 
        if(arg_name in ["break", "empty_space"]): 
            complete_easy_order.append(arg_name)
            complete_hard_order.append(arg_name)
        else:
            for plot_dict in plot_dicts:
                if(plot_dict["args"].arg_name == arg_name):    
                    if(plot_dict["args"].hard_maze): complete_hard_order.append(arg_name) ; hard_plot_dicts.append(plot_dict) ; hard = True
                    else:                            complete_easy_order.append(arg_name) ; easy_plot_dicts.append(plot_dict) ; easy = True
                    
    while(len(complete_easy_order) > 0 and complete_easy_order[0] in ["break", "empty_space"]): complete_easy_order.pop(0)
    while(len(complete_hard_order) > 0 and complete_hard_order[0] in ["break", "empty_space"]): complete_hard_order.pop(0)              
            
    return(plot_dicts, min_max_dict, (easy, complete_easy_order, easy_plot_dicts), (hard, complete_hard_order, hard_plot_dicts))
# %%
