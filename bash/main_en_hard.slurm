#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 48:00:00
#SBATCH --mem=50G

#SBATCH --ntasks=36
module load singularity
singularity exec maze.sif python pvrnn/main.py --comp deigo --arg_name en_hard --alpha None --curiosity prediction_error --maze_list "['t']" --epochs "[500]" --time_scales "[1]" --image_size 8 --max_steps 30 --min_speed 0 --max_speed 75 --better_reward "[(1,0),(1,10)]" --step_lim_punishment -2 --step_cost 0.99 --prediction_error_eta 0.5 --hidden_state_eta "[.5]" --agents_per_pos_list 36  --agents $agents_per_job --previous_agents $previous_agents
