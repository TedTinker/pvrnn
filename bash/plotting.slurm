#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 48:00:00
#SBATCH --mem=50G

module load singularity
singularity exec maze.sif python pvrnn/plotting.py --comp deigo --arg_title ___d_hard+d_hard_rand+break+e_hard+e_hard_rand+break+n_hard+n_hard_rand+break+en_hard+en_hard_rand+break+f_hard+f_hard_rand+break+ef_hard+ef_hard_rand___ --arg_name plotting
