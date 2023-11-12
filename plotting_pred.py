import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import torch
from torch.distributions import Normal
import torch.nn.functional as F

from utils import args, duration, load_dicts, print

print("name:\n{}".format(args.arg_name))
                                
                                                
        
def fix_image_size(image):
    while(len(image.shape) > 3): 
        image = image.squeeze(0)
    return(image[:,:,0:3])
    
def hard_plotting_pred(complete_order, plot_dicts):
    too_many_plot_dicts = len(plot_dicts) > 20
    epochs_maze_names = list(set(["_".join(key.split("_")[1:]) for key in plot_dicts[0]["pred_lists"].keys()]))
    epochs_maze_names.sort(key=lambda x: (int(x.split('_')[0]), x.split('_')[1]))
    agents = list(set([int(key.split("_")[0]) for key in plot_dicts[0]["pred_lists"].keys()])) ; agents.sort()
    first_arena_name = plot_dicts[0]["args"].maze_list[0] 
    episodes = len(plot_dicts[0]["pred_lists"]["1_0_{}".format(first_arena_name)])
    print("EPISODES:", episodes)
        
    for epoch_maze_name in epochs_maze_names:
        epoch, maze_name = epoch_maze_name.split("_")
        for agent in agents:
            print("HERE!", agent)
            for arg_name in complete_order:
                print(arg_name)
                if(arg_name in ["break", "empty_space"]): pass 
                else:
                    for plot_dict in plot_dicts:
                        print("\t", plot_dict["arg_name"])
                        if(plot_dict["arg_name"] == arg_name): 
                            pred_lists = plot_dict["pred_lists"]["{}_{}_{}".format(agent, epoch, maze_name)]
                            print(pred_lists); break 
                    for episode in range(episodes):
                        print("Episode", episode)
                        pred_list = pred_lists[episode]
                        rows = len(pred_list) ; columns = 3
                        fig, axs = plt.subplots(rows, columns, figsize = (columns * 2, rows * 2.5))
                        title = "Agent {}: Epoch {} (Maze {}), Episode {}".format( agent, epoch, maze_name, episode)
                        fig.suptitle(plot_dict["arg_title"] + "\n" + title, y = 1.1)
                        for row, (action_name, (rgbd, spe), (pred_rgbd_p, pred_spe_p), (pred_rgbd_q, pred_spe_q)) in enumerate(pred_list):
                            print(row)
                            for column in range(columns):
                                ax = axs[row, column] ; ax.axis("off")
                                if(row == 0 and column > 0): pass
                                else:
                                    steps_per_step = plot_dict["args"].steps_per_step
                                    # Actual obs
                                    if(column == 0):   
                                        rgbd = fix_image_size(rgbd)
                                        ax.imshow(rgbd)
                                        ax.set_title("Step {}\nAction: {}\nSpeed {}".format(row, action_name, steps_per_step*round(spe.item())), fontsize = 12)
                                    # ZP Sample
                                    elif(column == 1): 
                                        pred_rgbd_p = fix_image_size(pred_rgbd_p)
                                        ax.imshow(torch.sigmoid(pred_rgbd_p)) 
                                        ax.set_title("Prior\nSpeed {}".format(steps_per_step*round(pred_spe_p.item())), fontsize = 12)
                                    # ZQ Sample
                                    elif(column == 2): 
                                        pred_rgbd_q = fix_image_size(pred_rgbd_q)
                                        ax.imshow(torch.sigmoid(pred_rgbd_q)) 
                                        ax.set_title("Posterior\nSpeed {}".format(steps_per_step*round(pred_spe_q.item())), fontsize = 12)
                        print("Saving!")
                        plt.savefig("{}/{}.png".format(arg_name, title), format = "png", bbox_inches = "tight", dpi=300)
                        plt.close()
                        
        print("{}:\tDone with hard epoch {}.".format(duration(), epoch))
    


plot_dicts, min_max_dict, (easy, complete_easy_order, easy_plot_dicts), (hard, complete_hard_order, hard_plot_dicts) = load_dicts(args)    
if(hard): print("\nPlotting predictions in hard maze(s).\n") ; hard_plotting_pred(complete_hard_order, hard_plot_dicts)    
print("\nDuration: {}. Done!".format(duration()))