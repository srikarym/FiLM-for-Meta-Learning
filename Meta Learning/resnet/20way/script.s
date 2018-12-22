#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=ch_20
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=msy290@nyu.edu
#SBATCH --output=slurm_ch_20.out
cd /scratch/msy290/maml_resnet_20way/

module load cudnn/8.0v6.0
module load cuda/8.0.44
module load tensorflow/python3.6/1.3.0


python main.py --datasource=omniglot --metatrain_iterations=40000 --meta_batch_size=16 --update_batch_size=1 --num_classes=20 --update_lr=0.1  --num_updates=5 --logdir=logs/omniglot20way/  >> log_train.txt

python main.py --datasource=omniglot --metatrain_iterations=40000 --meta_batch_size=16 --update_batch_size=1 --num_classes=20 --update_lr=0.1  --num_updates=5 --logdir=logs/omniglot20way/ --train=False --test_set=True >> log_test.txt
