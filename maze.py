#%%
import numpy as np
import pybullet as p
from math import pi, degrees, sin, cos

from utils import default_args, print
from arena import Arena



import torch
from torchvision.transforms.functional import resize

class Hard_Maze:
    
    def __init__(self, arena_name, GUI = False, args = default_args):
        self.args = args
        self.name = arena_name
        self.maze = Arena(arena_name, GUI, args)
        self.begin()
        
    def begin(self):
        self.steps = 0 
        self.maze.begin()
        self.agent_pos, self.agent_yaw, self.agent_spe = self.maze.get_pos_yaw_spe()
        
    def put_agent_here(self, step, pos, yaw, spe):
        self.steps = step
        self.agent_pos = pos ; self.agent_yaw = yaw ; self.agent_spe = spe
        x, y = cos(yaw)*spe, sin(yaw)*spe
        self.maze.resetBaseVelocity(x, y)
        self.maze.resetBasePositionAndOrientation(pos, yaw)
        
    def obs(self):
        x, y = cos(self.agent_yaw), sin(self.agent_yaw)
        view_matrix = p.computeViewMatrix(
            cameraEyePosition = [self.agent_pos[0], self.agent_pos[1], .4], 
            cameraTargetPosition = [self.agent_pos[0] - x, self.agent_pos[1] - y, .4], 
            cameraUpVector = [0, 0, 1], physicsClientId = self.maze.physicsClient)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov = 90, aspect = 1, nearVal = .01, 
            farVal = 10, physicsClientId = self.maze.physicsClient)
        _, _, rgba, depth, _ = p.getCameraImage(
            width=self.args.image_size, height=self.args.image_size,
            projectionMatrix=proj_matrix, viewMatrix=view_matrix, shadow = 0,
            physicsClientId = self.maze.physicsClient)
        
        rgb = np.divide(rgba[:,:,:-1], 255)
        d = np.nan_to_num(np.expand_dims(depth, axis=-1), nan=1)
        if(d.max() == d.min()): pass
        else: d = (d.max() - d)/(d.max()-d.min())
        rgbd = np.concatenate([rgb, d], axis = -1)
        rgbd = torch.from_numpy(rgbd).float().unsqueeze(0)
        spe = torch.tensor(self.agent_spe).unsqueeze(0).unsqueeze(0)
        
        return(rgbd, spe)
    
    def change_velocity(self, yaw_change, speed, verbose = False):
        old_yaw = self.agent_yaw
        new_yaw = old_yaw + yaw_change
        new_yaw %= 2*pi
        self.maze.resetBasePositionAndOrientation((self.agent_pos[0], self.agent_pos[1], .5), new_yaw)
        
        old_speed = self.agent_spe
        x = -cos(new_yaw)*speed
        y = -sin(new_yaw)*speed
        self.maze.resetBaseVelocity(x, y)
        _, self.agent_yaw, _ = self.maze.get_pos_yaw_spe()
                
        if(verbose):
            print("\n\nOld yaw:\t{}\nChange:\t\t{}\nNew yaw:\t{}".format(
                round(degrees(old_yaw)) % 360, round(degrees(yaw_change)), round(degrees(new_yaw))))
            print("Old speed:\t{}\nNew speed:\t{}".format(old_speed, speed))
            #self.render(view = "body")  
            print("\n")
        
    def action(self, yaw, spe, verbose = True):
        self.steps += 1
        if((self.args.randomness > 0 or self.args.random_by_choice) and self.steps % self.args.random_steps == 0): self.maze.randomize()
        
        if(verbose): print("\n\nStep {}: yaw {}, spe {}.".format(self.steps, yaw, spe))
        yaw = -yaw * self.args.max_yaw_change
        yaw = [-self.args.max_yaw_change, self.args.max_yaw_change, yaw] ; yaw.sort() ; yaw = yaw[1]
        spe = self.args.min_speed + ((spe + 1)/2) * \
            (self.args.max_speed - self.args.min_speed)
        spe = [self.args.min_speed, self.args.max_speed, spe] ; spe.sort() ; spe = spe[1]
        if(verbose): print("updated: yaw {}, spe {}.".format(yaw, spe))
        action_name = "Yaw: {}. Speed: {}.".format(-1*round(degrees(yaw)), round(spe))
        
        for _ in range(self.args.steps_per_step):
            self.change_velocity(yaw/self.args.steps_per_step, spe/self.args.steps_per_step, verbose = verbose)
            p.stepSimulation(physicsClientId = self.maze.physicsClient)
            self.agent_pos, self.agent_yaw, self.agent_spe = self.maze.get_pos_yaw_spe()
            
        if(verbose): print("agent: pos {}, yaw {}, spe {}.".format(self.agent_pos, self.agent_yaw, self.agent_spe))
        
        end, which, reward = self.maze.end_collisions()
        if(reward > 0): reward *= self.args.step_cost ** self.steps
        if(verbose): print("end {}, which {}, reward {}".format(end, which, reward))
        
        col = self.maze.other_collisions()
        wall_punishment = self.args.wall_punishment if col else 0
        if(not end): end = self.steps >= self.args.max_steps
        exit = which != "NONE"
        if(end and not exit): reward += self.args.step_lim_punishment
        if(verbose): print("end {}, which {}, reward {}\n\n".format(end, which, reward + wall_punishment))

        return(reward, wall_punishment, which, end, action_name)
    
    
    
if __name__ == "__main__":        
    from random import random
    from time import sleep
    import matplotlib.pyplot as plt

    default_args.randomness = 0
    default_args.random_by_choice = True
    maze = Hard_Maze("2", True, default_args)
    done = False
    i = 0
    yaws = [0, 0, -1, 0, 0]
    speeds = [-1, -1, -1, -1, -1]
    while(done == False):
        #reward, wall_punishment, name, done, action_name = maze.action(random(), random(), verbose = True)
        reward, wall_punishment, name, done, action_name = maze.action(yaws[i], speeds[i], verbose = True)
        rgbd, spe = maze.obs()
        rgb = rgbd.squeeze(0)[:,:,0:3]
        d = rgbd.squeeze(0)[:,:,-1]
        plt.imshow(rgb)
        plt.show()
        plt.close()
        plt.imshow(d, cmap="gray")
        plt.show()
        plt.close()
        sleep(1)
        i += 1
# %%
