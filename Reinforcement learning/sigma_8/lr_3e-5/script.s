#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --time=160:00:00
#SBATCH --mem=60GB
#SBATCH --job-name=s8_e3-5
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=msy290@nyu.edu
#SBATCH --output=sigma_8_3e-5.out

source /scratch/msy290/env_tf/bin/activate

# Move to main directory
cd "/scratch/msy290/RL/film_sigma/sigma_8/lr_3e-5/"

# :'
# usage: ppo2ttifrutti_agent.py [-h]
#                               lr log_path [sigma_dim] [total_timesteps]  [nsteps]
#                               [nminibatches] [train] 
# [.] meant has default arguments here
# positional arguments:
#   lr               Learning rate
#   log_path         Log files saving path
#   total_timesteps  Total timesteps for the environment to run
#   nsteps           Number of steps
#   nminibatches     Number of minibatches
#   train            Is it training phase?
# 'sigma_dim         Dimension of sigma (4/8/16)

cd "metalearner"
# Training Part
python ppo2ttifrutti_agent.py 0.00003 '../logging/' 8 100000000

# :'usage: ppo2ttifrutti_agent.py [-h]
#                               val_idx log_path load_path [lr] [sigma_dim]
#                               [total_timesteps] [nsteps] [nminibatches]
# [.] meant has default arguments here
# positional arguments:
#   val_idx          Learning rate
#   log_path         Log files saving path
#   load_path        Load files from this path
#   lr               Learning rate
#   total_timesteps  Total timesteps for the environment to run
#   nsteps           Number of steps
#   nminibatches     Number of minibatches
# '

# Validation Part
cd "../fastlearner"

python ppo2ttifrutti_agent.py 1 '../logging/' '../logging/model/checkpoints/' 0.00003 8 1000000
python ppo2ttifrutti_agent.py 0 '../logging/' '../logging/model/checkpoints/' 0.00003 8 1000000
python ppo2ttifrutti_agent.py 2 '../logging/' '../logging/model/checkpoints/' 0.00003 8 1000000
python ppo2ttifrutti_agent.py 3 '../logging/' '../logging/model/checkpoints/' 0.00003 8 1000000
python ppo2ttifrutti_agent.py 4 '../logging/' '../logging/model/checkpoints/' 0.00003 8 1000000
python ppo2ttifrutti_agent.py 5 '../logging/' '../logging/model/checkpoints/' 0.00003 8 1000000
python ppo2ttifrutti_agent.py 6 '../logging/' '../logging/model/checkpoints/' 0.00003 8 1000000
python ppo2ttifrutti_agent.py 7 '../logging/' '../logging/model/checkpoints/' 0.00003 8 1000000
python ppo2ttifrutti_agent.py 8 '../logging/' '../logging/model/checkpoints/' 0.00003 8 1000000
python ppo2ttifrutti_agent.py 9 '../logging/' '../logging/model/checkpoints/' 0.00003 8 1000000
python ppo2ttifrutti_agent.py 10 '../logging/' '../logging/model/checkpoints/' 0.00003 8 1000000
