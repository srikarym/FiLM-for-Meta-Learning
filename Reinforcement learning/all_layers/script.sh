#!/bin/bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --time=160:00:00
#SBATCH --mem=60GB
#SBATCH --job-name=all_1e-4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cpk290@nyu.edu
#SBATCH --output=slurm_%j.out

module load python3/intel/3.6.3
# module load cudnn/9.0v7.3.0.29 cuda/9.0.176
source /home/cpk290/openai_env_1_cpu/bin/activate

# Move to main directory
# cd "/home/chandu/Desktop/courses/1st_semester/cv_project/retro_contest_agent_2"
cd "/scratch/cpk290/computer_vision/contest/all_layers/all_1e-4"

# :'
# usage: ppo2ttifrutti_agent.py [-h]
#                               lr log_path [total_timesteps] [nsteps]
#                               [nminibatches] [train]
# [.] meant has default arguments here
# positional arguments:
#   lr               Learning rate
#   log_path         Log files saving path
#   total_timesteps  Total timesteps for the environment to run
#   nsteps           Number of steps
#   nminibatches     Number of minibatches
#   train            Is it training phase?
# '

cd "metalearner"
# Training Part
python ppo2ttifrutti_agent.py 0.0001 '../logging/'

# :'usage: ppo2ttifrutti_agent.py [-h]
#                               val_idx log_path load_path [lr]
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

python ppo2ttifrutti_agent.py 0 '../logging/' '../logging/model/checkpoints/10000'
python ppo2ttifrutti_agent.py 1 '../logging/' '../logging/model/checkpoints/10000'
python ppo2ttifrutti_agent.py 2 '../logging/' '../logging/model/checkpoints/10000' 
python ppo2ttifrutti_agent.py 3 '../logging/' '../logging/model/checkpoints/10000' 
python ppo2ttifrutti_agent.py 4 '../logging/' '../logging/model/checkpoints/10000' 
python ppo2ttifrutti_agent.py 5 '../logging/' '../logging/model/checkpoints/10000' 
python ppo2ttifrutti_agent.py 6 '../logging/' '../logging/model/checkpoints/10000' 
python ppo2ttifrutti_agent.py 7 '../logging/' '../logging/model/checkpoints/10000' 
python ppo2ttifrutti_agent.py 8 '../logging/' '../logging/model/checkpoints/10000' 
python ppo2ttifrutti_agent.py 9 '../logging/' '../logging/model/checkpoints/10000' 
python ppo2ttifrutti_agent.py 10 '../logging/' '../logging/model/checkpoints/10000' 
