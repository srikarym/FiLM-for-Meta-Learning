#!/usr/bin/env python

import numpy as np
import gym
import gym_remote.client as grc
from retro_contest.local import make
from baselines.common.atari_wrappers import FrameStack
from baselines.bench import Monitor
from ppo2ttifrutti_agent import args
import cv2
cv2.ocl.setUseOpenCL(False)

class WarpFrame96(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 96x96."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 96
        self.height = 96
        self.observation_space = gym.spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]

class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()

class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.

    This is incredibly important and effects performance
    drastically.
    """
    def reward(self, reward):
        return reward * 0.01

class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """
    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs): # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info

def make_custom(stack=True, scale_rew=True):
    """
    Create an environment with some standard wrappers.
    """
    env = grc.RemoteEnv('tmp/sock')
    env = SonicDiscretizer(env)
    if scale_rew:
        env = RewardScaler(env)
    env = WarpFrame96(env)
    if stack:
        env = FrameStack(env, 4)
    env = AllowBacktracking(env)
    return env

def make_val(env_idx, stack=True, scale_rew=True):
    """
    Create an environment with some standard wrappers.
    """
    dicts = [
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'SpringYardZone.Act1'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'GreenHillZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'StarLightZone.Act3'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'ScrapBrainZone.Act1'},
        {'game': 'SonicTheHedgehog2-Genesis', 'state': 'MetropolisZone.Act3'},
        {'game': 'SonicTheHedgehog2-Genesis', 'state': 'HillTopZone.Act2'},
        {'game': 'SonicTheHedgehog2-Genesis', 'state': 'CasinoNightZone.Act2'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'LavaReefZone.Act1'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'FlyingBatteryZone.Act2'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'HydrocityZone.Act1'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'AngelIslandZone.Act2'}
    ]
    print(dicts[env_idx]['game'], dicts[env_idx]['state'], flush=True)
    env = make(game=dicts[env_idx]['game'], state=dicts[env_idx]['state'])#, bk2dir='/tmp')#, record='/tmp')
    log_dir = args.log_path+'val_'+str(env_idx)+'/log/'
    env = Monitor(env, log_dir, allow_early_resets=True)
    env = SonicDiscretizer(env)
    if scale_rew:
        env = RewardScaler(env)
    env = WarpFrame96(env)
    if stack:
        env = FrameStack(env, 4)
    env = AllowBacktracking(env)
    return env

def make_val_0(stack=True, scale_rew=True):
    return make_val(0, stack, scale_rew)

def make_val_1(stack=True, scale_rew=True):
    return make_val(1, stack, scale_rew)

def make_val_2(stack=True, scale_rew=True):
    return make_val(2, stack, scale_rew)

def make_val_3(stack=True, scale_rew=True):
    return make_val(3, stack, scale_rew)

def make_val_4(stack=True, scale_rew=True):
    return make_val(4, stack, scale_rew)

def make_val_5(stack=True, scale_rew=True):
    return make_val(5, stack, scale_rew)

def make_val_6(stack=True, scale_rew=True):
    return make_val(6, stack, scale_rew)

def make_val_7(stack=True, scale_rew=True):
    return make_val(7, stack, scale_rew)

def make_val_8(stack=True, scale_rew=True):
    return make_val(8, stack, scale_rew)

def make_val_9(stack=True, scale_rew=True):
    return make_val(9, stack, scale_rew)

def make_val_10(stack=True, scale_rew=True):
    return make_val(10, stack, scale_rew)