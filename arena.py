#%%
from random import choices, random, sample
import pandas as pd
import numpy as np
import pybullet as p
import cv2, os
from itertools import product
from math import pi, sin, cos

from utils import default_args, args, print

class Exit:
    def __init__(self, name, pos, rew):     # Position (Y, X)
        self.name = name ; self.pos = pos ; self.rew = rew
        
class Arena_Description:
    def __init__(self, start, exits, random_by_choice = []):
        self.start = start 
        self.exit_list = exits
        self.exits = pd.DataFrame(
            data = [[exit.name, exit.pos, exit.rew] for exit in exits],
            columns = ["Name", "Position", "Reward"])
        self.random_by_choice = random_by_choice
        
        
        
arena_dict = {
    "t.png" : Arena_Description(
        (3, 1),
        [Exit(  "LEFT",     (2,0), "default"),
        Exit(   "RIGHT",    (2,4), "better")],
        [(1, 0), (3, 0), (4, 1)]
        ),
    
    "1.png" : Arena_Description(
        (2,2), 
        [Exit(  "LEFT",    (1,0), "default"),
        Exit(   "RIGHT",    (1,4), "better")],
        [(0, 0), (0, 1), 
         (2, 0), (2, 1)] # Left side
        ),
    
    "2.png" : Arena_Description(
        (3,3), 
        [Exit(  "LEFT\nLEFT",   (4,1), "default"),
        Exit(   "LEFT\nRIGHT",  (0,1), "better"),
        Exit(   "RIGHT\nLEFT",  (0,5), "default"),
        Exit(   "RIGHT\nRIGHT", (4,5), "default")],
        [(3, 0), (3, 2), 
         (4, 0), (4, 2)] # Left Left. This one works, it's in the paper.
        #[(3, 4), (3, 6), (4, 4), (4, 6)] # Right Right. This one would be easier to explain.
        ),
    "3.png" : Arena_Description(
        (4,4), 
        [Exit(  "LEFT\nLEFT\nLEFT",    (6,3), "default"),
        Exit(   "LEFT\nLEFT\nRIGHT",   (6,1), "default"),
        Exit(   "LEFt\nRIGHT\nLEFT",   (0,1), "default"),
        Exit(   "LEFT\nRIGHT\nRIGHT",  (0,3), "default"),
        Exit(   "RIGHT\nLEFT\nLEFT",   (0,5), "better"),
        Exit(   "RIGHT\nLEFT\nRIGHT",  (0,7), "default"),
        Exit(   "RIGHT\nRIGHT\nLEFT",  (6,7), "default"),
        Exit(   "RIGHT\nRIGHT\nRIGHT", (6,5), "default")],
        [(4, 1), (4, 3), 
         (5, 0), 
         (6, 0), (6, 2)] # Left, Left, Right
        #[(0, 8), (1, 8)]
        )}



def get_physics(GUI, w, h):
    if(GUI):
        physicsClient = p.connect(p.GUI)
    else:   
        physicsClient = p.connect(p.DIRECT)
    p.resetDebugVisualizerCamera(1,90,-89,(w/2,h/2-.5,w), physicsClientId = physicsClient)
    p.setAdditionalSearchPath("pybullet_data/")
    return(physicsClient)

def enable_opengl():
    import pkgutil
    egl = pkgutil.get_loader('eglRenderer')
    import pybullet_data

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
    # print("plugin=", plugin)

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

photo_size = 2048
image_size = 2048

class Arena():
    def __init__(self, arena_name, GUI = False, args = default_args):
        #enable_opengl()
        self.args = args
        if(not arena_name.endswith(".png")): arena_name += ".png"
        self.start = arena_dict[arena_name].start
        self.exits = arena_dict[arena_name].exits
        arena_map = cv2.imread("arenas/" + arena_name)
        w, h, _ = arena_map.shape
        self.w, self.h = w, h
        self.physicsClient = get_physics(GUI, w, h)
        
        name = arena_name.split(".")[0]
        self.random_pos = []

        self.ends = {} ; self.colors = {} ; self.cube_locs = []
        for loc in ((x, y) for x in range(w) for y in range(h)):
            pos = [loc[0], loc[1], .5]
            if ((arena_map[loc] == [255]).all()):
                if (not self.exits.loc[self.exits["Position"] == loc].empty):
                    row = self.exits.loc[self.exits["Position"] == loc]
                    end_pos = ((pos[0] - .5, pos[0] + .5), (pos[1] - .5, pos[1] + .5))
                    self.ends[row["Name"].values[0]] = (end_pos, row["Reward"].values[0])
            else:
                ors = p.getQuaternionFromEuler([0, 0, 0])
                color = arena_map[loc][::-1] / 255
                color = np.append(color, 1)
                cube = p.loadURDF("cube.urdf", (pos[0], pos[1], pos[2]), ors, 
                                    useFixedBase=True, physicsClientId=self.physicsClient)
                self.colors[cube] = color
                self.cube_locs.append(loc)
        
            
        if(args.random_by_choice):
            self.random_pos = arena_dict[arena_name].random_by_choice
        else:
            self.random_pos = sample(self.cube_locs, k=int(len(self.cube_locs) * args.randomness))
        #print(args.randomness, args.random_by_choice)
        #print("\n\nRandom positions:", self.random_pos, "\n\n")
        
        for cube, color in self.colors.items():
            p.changeVisualShape(cube, -1, rgbaColor = color, physicsClientId = self.physicsClient)

        inherent_roll = pi/2
        inherent_pitch = 0
        yaw = 0
        spe = self.args.min_speed
        file = "ted_duck.urdf"
        pos = (self.start[0], self.start[1], .5)
        orn = p.getQuaternionFromEuler([inherent_roll, inherent_pitch, yaw])
        self.body_num = p.loadURDF(file, pos, orn,
                           globalScaling = self.args.body_size, 
                           physicsClientId = self.physicsClient)
        p.changeDynamics(self.body_num, 0, maxJointVelocity=10000)
        x, y = cos(yaw)*spe, sin(yaw)*spe
        self.resetBaseVelocity(x, y)
        p.changeVisualShape(self.body_num, -1, rgbaColor = [1,0,0,1], physicsClientId = self.physicsClient)
        
    def begin(self):
        yaw = 0
        spe = self.args.min_speed
        pos = (self.start[0], self.start[1], .5)
        x, y = cos(yaw)*spe, sin(yaw)*spe
        self.resetBaseVelocity(x, y)
        self.resetBasePositionAndOrientation(pos, yaw)
        if(self.args.randomness > 0): self.randomize()
        
    def get_pos_yaw_spe(self):
        pos, ors = p.getBasePositionAndOrientation(self.body_num, physicsClientId = self.physicsClient)
        yaw = p.getEulerFromQuaternion(ors)[-1]
        (x, y, _), _ = p.getBaseVelocity(self.body_num, physicsClientId = self.physicsClient)
        spe = (x**2 + y**2)**.5
        return(pos, yaw, spe)
    
    def resetBasePositionAndOrientation(self, pos, yaw):
        inherent_roll = pi/2
        inherent_pitch = 0
        orn = p.getQuaternionFromEuler([inherent_roll, inherent_pitch, yaw])
        p.resetBasePositionAndOrientation(self.body_num, pos, orn, physicsClientId = self.physicsClient)
        
    def resetBaseVelocity(self, x, y):    
        p.resetBaseVelocity(self.body_num, (x,y,0), (0,0,0), physicsClientId = self.physicsClient)
    
    def pos_in_box(self, box):
        (min_x, max_x), (min_y, max_y) = box 
        pos, _, _ = self.get_pos_yaw_spe()
        in_x = pos[0] >= min_x and pos[0] <= max_x 
        in_y = pos[1] >= min_y and pos[1] <= max_y 
        return(in_x and in_y)
    
    def end_collisions(self):
        col = False
        which = "NONE"
        reward = ((1, 0),)
        for end_name, (end, end_reward) in self.ends.items():
            if self.pos_in_box(end):
                col = True
                which = end_name
                reward = self.args.better_reward if end_reward == "better" else self.args.default_reward
        weights = [w for w, r in reward]
        rewards = [r for w, r in reward]
        reward = choices(rewards, weights = weights, k = 1)[0]
        return(col, which, reward)
    
    def other_collisions(self):
        col = False
        for cube in self.colors.keys():
            if 0 < len(p.getContactPoints(self.body_num, cube, physicsClientId = self.physicsClient)):
                col = True
        return(col)
    
    def in_random(self):
        random = False
        for x, y in self.random_pos:
            box = ((x - .5, x + .5), (y - .5, y + .5))
            if(self.pos_in_box(box)): random = True 
        return(random)
                
    def randomize(self):
        for cube in self.colors.keys():
            pos, _ = p.getBasePositionAndOrientation(cube, physicsClientId = self.physicsClient)
            if(pos[:-1] in self.random_pos):
                p.changeVisualShape(cube, -1, rgbaColor = [choices([0,1])[0], choices([0,1])[0], choices([0,1])[0], 1], physicsClientId = self.physicsClient)
            
    def stop(self):
        p.disconnect(self.physicsClient)
        
        
    def photo_from_above(self):
        view_matrix = p.computeViewMatrix(
            cameraEyePosition = [self.w/2, self.h/2 - .5, self.w], 
            cameraTargetPosition = [self.w/2, self.h/2 - .5, 0], 
            cameraUpVector = [1, 0, 0], physicsClientId = self.physicsClient)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov = 90, aspect = 1, nearVal = .01, 
            farVal = 100, physicsClientId = self.physicsClient)
        _, _, rgba, _, _ = p.getCameraImage(
            width=photo_size, height=photo_size,
            projectionMatrix=proj_matrix, viewMatrix=view_matrix, shadow = 0,
            physicsClientId = self.physicsClient)
        
        rgb = np.divide(rgba[:,:,:-1], 255)
        photo = cv2.resize(rgb, (image_size, image_size))
        photo = np.flip(photo, axis = 1)
        view_matrix = np.reshape(view_matrix, (4, 4))
        proj_matrix = np.reshape(proj_matrix, (4, 4))
        return(photo, view_matrix, proj_matrix)
        
        
        
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    def fix_x_y(x, y):
        return(x/10, -y/10)
        
    
    for maze_name in arena_dict.keys():
        
        #print(maze_name)
        arena = Arena(maze_name)
        min_x, min_y, max_x, max_y = float("inf"), float("inf"), -float("inf"), -float("inf")
        for (x, y) in arena.cube_locs:
            x, y = fix_x_y(x, y)
            if(x < min_x): min_x = x 
            if(x > max_x): max_x = x 
            if(y < min_y): min_y = y 
            if(y > max_y): max_y = y 
        w, h = arena.w, arena.h
        photo, view_matrix, proj_matrix = arena.photo_from_above()
        
        plt.figure(figsize = (2*w,2*h))
        plt.imshow(photo, origin = "lower", zorder = 1)#, extent=[min_x, max_x, min_y, max_y])
        plt.axis("off")
        #plt.axis([min_x, max_x, min_y, max_y])
        plt.savefig("arenas/{}_map.png".format(maze_name[:-4]), bbox_inches = "tight")
        plt.show()
        plt.close()
        
        plt.figure(figsize = (2*w,2*h))
        exits = arena_dict[maze_name].exit_list
        for exit in exits:
            y, x = exit.pos
            x, y = fix_x_y(x, y)
            if(exit.rew == "default"): text = "X" ; fontsize = 80 
            if(exit.rew == "better"):  text = "✓" ; fontsize = 100 
            plt.text(x, y, text, fontsize = fontsize, ha='center', va='center', zorder = 4)
        traps = arena_dict[maze_name].random_by_choice
        text = "?" ; fontsize = 80
        for y, x in traps:
            x, y = fix_x_y(x, y)
            plt.text(x, y, text, fontsize = fontsize, ha='center', va='center', zorder = 4)
        plt.axis("off")
        plt.autoscale(enable=False)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis([min_x, max_x, min_y, max_y])
        plt.savefig("arenas/{}_text.png".format(maze_name[:-4]), transparent = True, bbox_inches = "tight")
        plt.show()
        plt.close()
        
        arena.stop()

# %%
