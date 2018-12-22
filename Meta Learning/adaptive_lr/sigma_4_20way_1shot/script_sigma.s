#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=4_20_sigma
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cpk290@nyu.edu
#SBATCH --output=slurm_%j.out

cd /scratch/cpk290/computer_vision/maml_film/maml_less_grad_sigma_4_20way_1shot
module load cudnn/8.0v6.0
module load cuda/8.0.44
module load tensorflow/python3.6/1.3.0
python main.py --sigma_dim=4 --datasource=omniglot --metatrain_iterations=40000 --meta_batch_size=16 --update_batch_size=1 --num_classes=20 --update_lr=0.1 --num_updates=5 --logdir=logs/omniglot/ |& tee -a log_omniglot5way_train.txt
python main.py --train=False --test_set=True --sigma_dim=4 --datasource=omniglot --metatrain_iterations=40000 --meta_batch_size=16 --update_batch_size=1 --num_classes=20 --update_lr=0.1 --num_updates=5 --logdir=logs/omniglot/ 
