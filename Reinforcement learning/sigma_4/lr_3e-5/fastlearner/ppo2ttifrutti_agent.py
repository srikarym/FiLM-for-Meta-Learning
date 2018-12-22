#!/usr/bin/env python

"""
Train an agent on Sonic using PPO2ttifrutti, a variant of OpenAI PPO2 baseline.
"""

import tensorflow as tf
import numpy as np
import gym
import gym_remote.exceptions as gre
import math
import os

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

import ppo2ttifrutti
import ppo2ttifrutti_policies as policies

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("val_idx", type=int,
            help='Learning rate')

parser.add_argument("log_path", type=str,
            help='Log files saving path')

parser.add_argument("load_path", type=str,
            help='Load files from this path')

parser.add_argument("lr", type=float, default=7.5e-5, nargs='?',
            help='Learning rate')

parser.add_argument("sigma_dim",type=int,default=8, nargs='?',
            help='dimension of sigma?')

# parser.add_argument("n_envs", type=int, default=47,
#           help='Number of environments')
parser.add_argument("total_timesteps", type=int, default=int(1e7), nargs='?',
            help='Total timesteps for the environment to run')

parser.add_argument("nsteps", type=int, default=2048, nargs='?',
            help='Number of steps')

parser.add_argument("nminibatches", type=int, default=16, nargs='?',
            help='Number of minibatches')




# parser.add_argument("train",type=bool,default=False, nargs='?',
#             help='Is it training phase?')
args = parser.parse_args()

import ppo2ttifrutti_sonic_env as env

val_count = 11
nenvs = 47
stack = True
scale_rew = True
def main():
    """Run PPO until the environment throws an exception."""
    config = tf.ConfigProto()
    #os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    lis = [env.make_val_0, env.make_val_1,env.make_val_2,env.make_val_3,env.make_val_4,env.make_val_5,env.make_val_6,env.make_val_7,env.make_val_8,env.make_val_9,env.make_val_10]
    i = args.val_idx
    # Creating directory if necessary
    val_path = args.log_path+'val_'+str(i)+'/'
    if not os.path.isdir(val_path):
        os.mkdir(val_path)
        if not os.path.isdir(val_path+'log/'):
            os.mkdir(val_path+'log/')
        if not os.path.isdir(val_path+'model/'):
            os.mkdir(val_path+'model/')

    if not os.path.isdir(val_path+'log/'):
        os.mkdir(val_path+'log/')
    if not os.path.isdir(val_path+'model/'):
        os.mkdir(val_path+'model/')
    
    # Create session
    with tf.Session(config=config):
        # Take more timesteps than we need to be sure that
        # we stop due to an exception.
        ppo2ttifrutti.learn(policy=policies.CnnPolicy,
                            env=DummyVecEnv([lis[i]]),
                            nsteps=args.nsteps,
                            nminibatches=args.nminibatches,
                            lam=0.95,
                            gamma=0.99,
                            noptepochs=4,
                            log_interval=1,
                            ent_coef=0.01,
                            lr=lambda _: args.lr,
                            cliprange=lambda _: 0.1,
                            total_timesteps=args.total_timesteps,
                            load_path=tf.train.latest_checkpoint(args.load_path)[:-3],
                            log_path=val_path)
                            # train=args.train)

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
