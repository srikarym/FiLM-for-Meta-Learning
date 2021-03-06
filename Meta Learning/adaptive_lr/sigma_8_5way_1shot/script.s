#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=sig_8_5
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=msy290@nyu.edu
#SBATCH --output=slurm_8_5.out
cd /scratch/msy290/maml_less_grad_adaptive_lr/sigma_8_5way_1shot/

module load cudnn/8.0v6.0
module load cuda/8.0.44
module load tensorflow/python3.6/1.3.0



python main.py --datasource=omniglot --metatrain_iterations=40000 --meta_batch_size=32 --update_batch_size=1 --num_classes=5 --update_lr=0.4 --sigma_dim=8  --num_updates=1 --logdir=logs/omniglot5way/ --train=False --test_set=True
