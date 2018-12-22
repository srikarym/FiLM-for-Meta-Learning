#!/usr/bin/env python

"""
Train an agent on Sonic using PPO2ttifrutti, a variant of OpenAI PPO2 baseline.
"""

import tensorflow as tf
import numpy as np
import gym
import gym_remote.exceptions as gre
import os
import math

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

import ppo2ttifrutti
import ppo2ttifrutti_policies as policies

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("lr", type=float, 
            help='Learning rate')

parser.add_argument("log_path", type=str,
            help='Log files saving path')

# parser.add_argument("n_envs", type=int, default=47,
#           help='Number of environments')
parser.add_argument("sigma_dim",type=int,default=8, nargs='?',
            help='dimension of sigma?')
            
parser.add_argument("total_timesteps", type=int, default=int(1e9), nargs='?',
            help='Total timesteps for the environment to run')

parser.add_argument("nsteps", type=int, default=2048, nargs='?',
            help='Number of steps')

parser.add_argument("nminibatches", type=int, default=16, nargs='?',
            help='Number of minibatches')

parser.add_argument("train",type=bool,default=True, nargs='?',
            help='Is it training phase?')



args = parser.parse_args()

import ppo2ttifrutti_sonic_env as env

nenvs = 47

def main():
    """Run PPO until the environment throws an exception."""
    config = tf.ConfigProto()
    #os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    config.gpu_options.allow_growth = True # pylint: disable=E1101

    # Creating directory if necessary
    train_path = args.log_path
    if not os.path.isdir(train_path):
        os.mkdir(train_path)
        if not os.path.isdir(train_path+'log/'):
            os.mkdir(train_path+'log/')
        if not os.path.isdir(train_path+'model/'):
            os.mkdir(train_path+'model/')

    if not os.path.isdir(train_path+'log/'):
        os.mkdir(train_path+'log/')
    if not os.path.isdir(train_path+'model/'):
        os.mkdir(train_path+'model/')

    with tf.Session(config=config):
        # Take more timesteps than we need to be sure that
        # we stop due to an exception.
        ppo2ttifrutti.learn(policy=policies.CnnPolicy,
                            env=SubprocVecEnv([env.make_train_0, env.make_train_1, env.make_train_2, env.make_train_3, env.make_train_4, env.make_train_5, env.make_train_6, env.make_train_7, env.make_train_8, env.make_train_9, env.make_train_10, env.make_train_11, env.make_train_12, env.make_train_13, env.make_train_14, env.make_train_15, env.make_train_16, env.make_train_17, env.make_train_18, env.make_train_19, env.make_train_20, env.make_train_21, env.make_train_22, env.make_train_23, env.make_train_24, env.make_train_25, env.make_train_26, env.make_train_27, env.make_train_28, env.make_train_29, env.make_train_30, env.make_train_31, env.make_train_32, env.make_train_33, env.make_train_34, env.make_train_35, env.make_train_36, env.make_train_37, env.make_train_38, env.make_train_39, env.make_train_40, env.make_train_41, env.make_train_42, env.make_train_43, env.make_train_44, env.make_train_45, env.make_train_46]),
                            # env=SubprocVecEnv([env.make_train_0, env.make_train_1, env.make_train_2, env.make_train_3, env.make_train_4, env.make_train_5, env.make_train_6]),
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
                            save_interval=25,
                            log_path = args.log_path,
                            train=args.train)

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
