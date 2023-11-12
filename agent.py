#%%

import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import torch.optim as optim

import numpy as np
from math import log
from itertools import accumulate
from copy import deepcopy

from utils import default_args, detach_list, dkl, print
from maze import Hard_Maze
from buffer import RecurrentReplayBuffer
from models import Forward, Actor, Critic

action_size = 2



class Agent:
    
    def __init__(self, i, args = default_args):
        
        self.start_time = None
        
        self.agent_num = i
        self.args = args
        self.episodes = 0 ; self.epochs = 0 ; self.steps = 0
        self.maze_name = self.args.maze_list[0]
        self.maze = Hard_Maze(self.maze_name, args = args)
        
        self.target_entropy = args.target_entropy # -dim(A)
        self.alpha = 1
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha_opt = optim.Adam(params=[self.log_alpha], lr=args.alpha_lr) 
        
        self.naive_eta = 1
        self.log_naive_eta = torch.tensor([0.0], requires_grad=True)
        self.naive_eta_opt = optim.Adam(params=[self.log_naive_eta], lr=args.alpha_lr) 
        
        self.free_eta = [1] * len(args.time_scales)
        self.log_free_eta = [torch.tensor([0.0], requires_grad=True)] * len(args.time_scales)
        self.free_eta_opt = [optim.Adam(params=[self.log_free_eta[layer]], lr=args.alpha_lr) for layer in range(len(args.time_scales))]
        
        self.forward = Forward(args)
        self.forward_opt = optim.Adam(self.forward.parameters(), lr=args.forward_lr)
                           
        self.actor = Actor(args)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=args.actor_lr) 
        
        self.critic1 = Critic(args)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=args.critic_lr)
        self.critic1_target = Critic(args)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2 = Critic(args)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=args.critic_lr)
        self.critic2_target = Critic(args)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        self.train()
        
        self.memory = RecurrentReplayBuffer(args)
        self.plot_dict = {
            "args" : args,
            "arg_title" : args.arg_title,
            "arg_name" : args.arg_name,
            "pred_lists" : {}, "pos_lists" : {}, 
            "agent_lists" : {"forward" : Forward, "actor" : Actor, "critic" : Critic},
            "rewards" : [], "spot_names" : [], "steps" : [],
            "accuracy" : [], "complexity" : [],
            "alpha" : [], "actor" : [], 
            "critic_1" : [], "critic_2" : [], 
            "extrinsic" : [], "intrinsic_curiosity" : [], 
            "intrinsic_entropy" : [], 
            "naive" : [], "free" : [[] for _ in range(args.layers)]}
        
        
        
    def training(self, q):
        
        self.pred_episodes()
        self.pos_episodes()
        self.save_agent()
        while(True):
            cumulative_epochs = 0
            prev_maze_name = self.maze_name
            for j, epochs in enumerate(self.args.epochs): 
                cumulative_epochs += epochs
                if(self.epochs < cumulative_epochs): self.maze_name = self.args.maze_list[j] ; break
            if(prev_maze_name != self.maze_name): 
                self.pred_episodes()
                self.pos_episodes()
                self.maze.maze.stop()
                self.maze = Hard_Maze(self.maze_name, args = self.args)
                self.pred_episodes()
                self.pos_episodes()
            self.training_episode()
            percent_done = str(self.epochs / sum(self.args.epochs))
            q.put((self.agent_num, percent_done))
            if(self.epochs >= sum(self.args.epochs)): break
            if(self.epochs % self.args.epochs_per_pred_list == 0): self.pred_episodes()
            if(self.epochs % self.args.epochs_per_pos_list == 0): self.pos_episodes()
            if(self.epochs % self.args.epochs_per_agent_list == 0): self.save_agent()
        self.plot_dict["rewards"] = list(accumulate(self.plot_dict["rewards"]))
        self.pred_episodes()
        self.pos_episodes()
        self.save_agent()
        
        self.min_max_dict = {key : [] for key in self.plot_dict.keys()}
        self.min_max_dict["free"] = [] * self.args.layers
        for key in self.min_max_dict.keys():
            if(not key in ["args", "arg_title", "arg_name", "pred_lists", "pos_lists", "agent_lists", "spot_names", "steps"]):
                if(key == "free"):
                    for l in self.plot_dict[key]:
                        minimum = None ; maximum = None 
                        l = deepcopy(l)
                        l = [_ for _ in l if _ != None]
                        if(l != []):
                            if(  minimum == None):  minimum = min(l)
                            elif(minimum > min(l)): minimum = min(l)
                            if(  maximum == None):  maximum = max(l) 
                            elif(maximum < max(l)): maximum = max(l)
                        self.min_max_dict[key].append((minimum, maximum))
                else:
                    minimum = None ; maximum = None 
                    l = self.plot_dict[key]
                    l = deepcopy(l)
                    l = [_ for _ in l if _ != None]
                    if(l != []):
                        if(  minimum == None):  minimum = min(l)
                        elif(minimum > min(l)): minimum = min(l)
                        if(  maximum == None):  maximum = max(l) 
                        elif(maximum < max(l)): maximum = max(l)
                    self.min_max_dict[key] = (minimum, maximum)
                
                
                
    def save_agent(self):
        if(self.args.agents_per_agent_list != -1 and self.agent_num > self.args.agents_per_agent_list): return
        self.plot_dict["agent_lists"]["{}_{}".format(self.agent_num, self.epochs)] = deepcopy(self.state_dict())
    
    
    
    def step_in_episode(self, prev_action, hq_m1, push, verbose):
        with torch.no_grad():
            o, s = self.maze.obs()
            _, _, hp = self.forward.p(prev_action, hq_m1)
            _, _, hq = self.forward.q(prev_action, o, s, hq_m1)
            a, _, _ = self.actor(hq) 
            action = torch.flatten(a).tolist()
            r, wall_punishment, spot_name, done, action_name = self.maze.action(action[0], action[1], verbose)
            no, ns = self.maze.obs()
            if(push): 
                if(done and self.args.retroactive_reward): 
                    for i in range(self.memory.time_ptr):
                        retro_reward = r if r <= 0 else r * self.args.retro_step_cost ** (self.memory.time_ptr - i)
                        self.memory.r[self.memory.episode_ptr, i] += retro_reward
                self.memory.push(o, s, a, r + wall_punishment, no, ns, done, done)
        return(a, hp, hq, r + wall_punishment, spot_name, done, action_name)
            
            
            
    def pred_episodes(self):
        with torch.no_grad():
            if(self.args.agents_per_pred_list != -1 and self.agent_num > self.args.agents_per_pred_list): return
            pred_lists = []
            for episode in range(self.args.episodes_in_pred_list):
                done = False        
                prev_action = torch.zeros((1, 1, 2))     
                hq = [torch.zeros((1, 1, self.args.hidden_size))] * self.args.layers 
                self.maze.begin()
                
                o, s = self.maze.obs()
                pred_list = [(None, (o, s), (None, None), (None, None))]
                for step in range(self.args.max_steps):
                    if(not done): 
                        a, hp_p1, hq_p1, _, _, done, action_name = self.step_in_episode(prev_action, hq, push = False, verbose = False)
                        no, ns = self.maze.obs()
                        pred_rgbd_p, pred_speed_p = self.forward.predict(a, hp_p1) 
                        pred_rgbd_q, pred_speed_q = self.forward.predict(a, hq_p1) 
                        pred_list.append((
                            action_name, (no, ns), (pred_rgbd_p, pred_speed_p), (pred_rgbd_q, pred_speed_q)))
                        prev_action = a ; hq = hq_p1 
                pred_lists.append(pred_list)
            self.plot_dict["pred_lists"]["{}_{}_{}".format(self.agent_num, self.epochs, self.maze.name)] = pred_lists
    


    def pos_episodes(self):
        if(self.args.agents_per_pos_list != -1 and self.agent_num > self.args.agents_per_pos_list): return
        pos_lists = []
        for episode in range(self.args.episodes_in_pos_list):
            done = False 
            prev_action = torch.zeros((1, 1, 2))  
            hq = [torch.zeros((1, 1, self.args.hidden_size))] * self.args.layers 
            self.maze.begin()
            pos_list = [self.maze_name, self.maze.maze.get_pos_yaw_spe()[0]]
            for step in range(self.args.max_steps):
                if(not done): prev_action, hp, hq, _, _, done, _ = self.step_in_episode(prev_action, hq, push = False, verbose = False)
                pos_list.append(self.maze.maze.get_pos_yaw_spe()[0])
            pos_lists.append(pos_list)
        self.plot_dict["pos_lists"]["{}_{}_{}".format(self.agent_num, self.epochs, self.maze.name)] = pos_lists
    
    
    
    def training_episode(self, push = True, verbose = False):
        done = False ; steps = 0 ; cumulative_r = 0
        prev_action = torch.zeros((1, 1, 2))
        hq = None
        
        self.maze.begin()
        if(verbose): print("\n\n\n\n\nSTART!\n")
        
        for step in range(self.args.max_steps):
            self.steps += 1 
            if(not done):
                steps += 1
                prev_action, hp, hq, r, spot_name, done, _ = self.step_in_episode(prev_action, hq, push, verbose)
                cumulative_r += r
                
            if(self.steps % self.args.steps_per_epoch == 0):
                #print("episodes: {}. epochs: {}. steps: {}.".format(self.episodes, self.epochs, self.steps))
                plot_data = self.epoch(batch_size = self.args.batch_size)
                if(plot_data == False): pass
                else:
                    l, e, ic, ie, naive, free = plot_data
                    if(self.epochs == 1 or self.epochs >= sum(self.args.epochs) or self.epochs % self.args.keep_data == 0):
                        self.plot_dict["accuracy"].append(l[0][0])
                        self.plot_dict["complexity"].append(l[0][1])
                        self.plot_dict["alpha"].append(l[0][2])
                        self.plot_dict["actor"].append(l[0][3])
                        self.plot_dict["critic_1"].append(l[0][4])
                        self.plot_dict["critic_2"].append(l[0][5])
                        self.plot_dict["extrinsic"].append(e)
                        self.plot_dict["intrinsic_curiosity"].append(ic)
                        self.plot_dict["intrinsic_entropy"].append(ie)
                        self.plot_dict["naive"].append(naive)
                        for layer, f in enumerate(free):
                            self.plot_dict["free"][layer].append(f)    
        
        self.plot_dict["steps"].append(steps)
        self.plot_dict["rewards"].append(r)
        self.plot_dict["spot_names"].append(spot_name)
        self.episodes += 1
    
    
    
    def epoch(self, batch_size):
                                
        batch = self.memory.sample(batch_size)
        if(batch == False): return(False)
                
        self.epochs += 1

        rgbd, spe, actions, rewards, dones, masks = batch
        actions = torch.cat([torch.zeros(actions[:,0].unsqueeze(1).shape), actions], dim = 1)    
        all_masks = torch.cat([torch.ones(masks.shape[0], 1, 1), masks], dim = 1)   
        episodes = rewards.shape[0] ; steps = rewards.shape[1] 
        
        #print("\n\n")
        #print("{}. rgbd: {}. spe: {}. actions: {}. rewards: {}. dones: {}. masks: {}.".format(
        #    self.agent_num, rgbd.shape, spe.shape, actions.shape, rewards.shape, dones.shape, masks.shape))
        #print("\n\n")
        
                
        
        # Train forward
        (zp_mu_lists, zp_std_lists,                                        hp_lists), \
        (zq_mu_lists, zq_std_lists, zq_rgbd_pred_list, zq_speed_pred_list, hq_lists) = self.forward(actions, rgbd[:,:-1], spe[:,:-1])
        full_h_list = [h for h in hq_lists] ; h_list = [h[:, :-1] for h in hq_lists]
                
        zp_mu_list  = [torch.cat([zp_mu[layer]  for zp_mu  in zp_mu_lists],  dim = 1) for layer in range(self.args.layers)]
        zp_std_list = [torch.cat([zp_std[layer] for zp_std in zp_std_lists], dim = 1) for layer in range(self.args.layers)]
        zq_mu_list  = [torch.cat([zq_mu[layer]  for zq_mu  in zq_mu_lists],  dim = 1) for layer in range(self.args.layers)]
        zq_std_list = [torch.cat([zq_std[layer] for zq_std in zq_std_lists], dim = 1) for layer in range(self.args.layers)]
                
        pred_rgbd = torch.cat(zq_rgbd_pred_list,  dim = 1)
        pred_spe  = torch.cat(zq_speed_pred_list, dim = 1)

        image_loss = F.binary_cross_entropy_with_logits(pred_rgbd, rgbd[:,1:], reduction = "none").mean((-1,-2,-3)).unsqueeze(-1) * masks # all_masks
        speed_loss = self.args.speed_scalar * F.mse_loss(pred_spe, spe[:,1:],  reduction = "none").mean(-1).unsqueeze(-1) * masks # all_masks
        accuracy_for_naive = image_loss + speed_loss
        accuracy           = accuracy_for_naive.mean()
        #accuracy_for_naive = accuracy_for_naive[:,1:]
        
        complexity_for_free = [dkl(zq_mu, zq_std, zp_mu, zp_std).mean(-1).unsqueeze(-1) * masks for (zq_mu, zq_std, zp_mu, zp_std) in zip(zq_mu_list, zq_std_list, zp_mu_list, zp_std_list)]
        complexity          = sum([self.args.beta[layer] * complexity_for_free[layer].mean() for layer in range(self.args.layers)])        
                        
        self.forward_opt.zero_grad()
        (accuracy + complexity).backward()
        self.forward_opt.step()
        
        if(self.args.beta == 0): complexity = None
                        
                        
        
        # Get curiosity                  
        if(self.args.dkl_max != None):
            complexity_for_free = [torch.clamp(c, min = 0, max = self.args.dkl_max) for c in complexity_for_free]
            #complexity_for_free = torch.tanh(complexity_for_free) # Consider other clamps!
        naive_curiosity = accuracy_for_naive * (self.args.naive_eta if self.args.naive_eta != None else self.naive_eta)
        free_curiosities = [complexity_for_free[layer] * (self.args.free_eta[layer] if self.args.free_eta[layer] != None else self.free_eta[layer]) for layer in range(self.args.layers)]
        free_curiosity = sum(free_curiosities)
        if(self.args.curiosity == "naive"):  curiosity = naive_curiosity
        elif(self.args.curiosity == "free"): curiosity = free_curiosity
        else:                                curiosity = torch.zeros(rewards.shape)
        extrinsic = torch.mean(rewards).item()
        intrinsic_curiosity = curiosity.mean().item()
        rewards += curiosity
                        
        
                
        # Train critics
        with torch.no_grad():
            new_actions, log_pis_next, _ = self.actor(detach_list(full_h_list))
            Q_target1_next, _ = self.critic1_target(rgbd, spe, new_actions)
            Q_target2_next, _ = self.critic2_target(rgbd, spe, new_actions)
            log_pis_next = log_pis_next[:,1:]
            Q_target_next = torch.min(Q_target1_next, Q_target2_next)
            Q_target_next = Q_target_next[:,1:]
            if self.args.alpha == None: Q_targets = rewards + (self.args.GAMMA * (1 - dones) * (Q_target_next - self.alpha * log_pis_next))
            else:                       Q_targets = rewards + (self.args.GAMMA * (1 - dones) * (Q_target_next - self.args.alpha * log_pis_next))
        
        Q_1, _ = self.critic1(rgbd[:,:-1], spe[:,:-1], actions[:,1:])
        critic1_loss = 0.5*F.mse_loss(Q_1*masks, Q_targets*masks)
        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        self.critic1_opt.step()
        
        Q_2, _ = self.critic2(rgbd[:,:-1], spe[:,:-1], actions[:,1:])
        critic2_loss = 0.5*F.mse_loss(Q_2*masks, Q_targets*masks)
        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()
        
        self.soft_update(self.critic1, self.critic1_target, self.args.tau)
        self.soft_update(self.critic2, self.critic2_target, self.args.tau)
                        
        
        
        # Train alpha
        if self.args.alpha == None:
            _, log_pis, _ = self.actor(detach_list(h_list))
            alpha_loss = -(self.log_alpha * (log_pis + self.target_entropy))*masks
            alpha_loss = alpha_loss.mean() / masks.mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = torch.exp(self.log_alpha) 
        else:
            alpha_loss = None
            
            
            
        """# Train eta
        if(self.args.curiosity == "naive"):
            eta_losses = []
            if(self.args.naive_eta == None):
                eta_loss = -(self.log_naive_eta * (naive_curiosity + self.args.target_naive_curiosity))*masks
                eta_loss = eta_loss.mean() / masks.mean()
                self.naive_eta_opt.zero_grad()
                eta_loss.backward()
                self.naive_eta_opt.step()
                self.naive_eta = torch.exp(self.log_naive_eta) 
            else:
                eta_loss = None
            eta_losses.append(eta_loss)
        elif(self.args.curiosity == "free"):
            eta_losses = []
            for layer, eta in enumerate(self.args.free_eta):
                if(eta == None):
                    eta_loss = -(self.log_free_eta[layer] * (free_curiosities[layer] + self.args.target_free_curiosity[layer]))*masks
                    eta_loss = eta_loss.mean() / masks.mean()
                    self.free_eta_opt[layer].zero_grad()
                    eta_loss.backward()
                    self.free_eta_opt[layer].step()
                    self.free_eta[layer] = torch.exp(self.log_free_eta[layer]) 
                else:
                    eta_loss = None
                eta_losses.append(eta_loss)"""
                
        
        # Consider training beta and gamma! 
                                    
            
        
        # Train actor
        if self.epochs % self.args.d == 0:
            if self.args.alpha == None: alpha = self.alpha 
            else:                       alpha = self.args.alpha
            new_actions, log_pis, _ = self.actor(detach_list(h_list))

            if self.args.action_prior == "normal":
                loc = torch.zeros(action_size, dtype=torch.float64)
                scale_tril = torch.tensor([[1, 0], [1, 1]], dtype=torch.float64)
                policy_prior = MultivariateNormal(loc=loc, scale_tril=scale_tril)
                policy_prior_log_prrgbd = policy_prior.log_prob(new_actions).unsqueeze(-1)
            elif self.args.action_prior == "uniform":
                policy_prior_log_prrgbd = 0.0
            Q_1, _ = self.critic1(rgbd[:,:-1], spe[:,:-1], new_actions)
            Q_2, _ = self.critic2(rgbd[:,:-1], spe[:,:-1], new_actions)
            Q = torch.min(Q_1, Q_2).mean(-1).unsqueeze(-1)
            intrinsic_entropy = torch.mean((alpha * log_pis)*masks).item()
            actor_loss = (alpha * log_pis - policy_prior_log_prrgbd - Q)*masks
            actor_loss = actor_loss.mean() / masks.mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            
        else:
            intrinsic_entropy = None
            actor_loss = None
                                
                                
                                
        if(accuracy != None):   accuracy = accuracy.item()
        if(complexity != None): complexity = complexity.item()
        if(alpha_loss != None): alpha_loss = alpha_loss.item()
        if(actor_loss != None): actor_loss = actor_loss.item()
        if(critic1_loss != None): 
            critic1_loss = critic1_loss.item()
            critic1_loss = log(critic1_loss) if critic1_loss > 0 else critic1_loss
        if(critic2_loss != None): 
            critic2_loss = critic2_loss.item()
            critic2_loss = log(critic2_loss) if critic2_loss > 0 else critic2_loss
        losses = np.array([[accuracy, complexity, alpha_loss, actor_loss, critic1_loss, critic2_loss]])
        
        naive_curiosity = naive_curiosity.mean().item()
        free_curiosities = [free_curiosity.mean().item() for free_curiosity in free_curiosities]
        free_curiosities = [free_curiosity for free_curiosity in free_curiosities]
        
        return(losses, extrinsic, intrinsic_curiosity, intrinsic_entropy, naive_curiosity, free_curiosities)
    
    
                     
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def state_dict(self):
        return(
            self.forward.state_dict(),
            self.actor.state_dict(),
            self.critic1.state_dict(),
            self.critic1_target.state_dict(),
            self.critic2.state_dict(),
            self.critic2_target.state_dict())

    def load_state_dict(self, state_dict):
        self.forward.load_state_dict(state_dict[0])
        self.actor.load_state_dict(state_dict[1])
        self.critic1.load_state_dict(state_dict[2])
        self.critic1_target.load_state_dict(state_dict[3])
        self.critic2.load_state_dict(state_dict[4])
        self.critic2_target.load_state_dict(state_dict[5])
        self.memory = RecurrentReplayBuffer(self.args)

    def eval(self):
        self.forward.eval()
        self.actor.eval()
        self.critic1.eval()
        self.critic1_target.eval()
        self.critic2.eval()
        self.critic2_target.eval()

    def train(self):
        self.forward.train()
        self.actor.train()
        self.critic1.train()
        self.critic1_target.train()
        self.critic2.train()
        self.critic2_target.train()
        
        
        
if __name__ == "__main__":
    agent = Agent(0)
# %%
