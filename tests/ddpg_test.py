import gym
import numpy as np

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
from stable_baselines.common.vec_env import VecVideoRecorder, DummyVecEnv
from gym.wrappers import FlattenDictWrapper
import os.path as osp

expDir = '/home/shivanik/lab/pointExp/state/'
verbose = 1
nIter = 5e6
name = 'ddpg_1_%s' %np.format_float_scientific(nIter)
logger = osp.join(expDir, name, 'logs')
video_folder = osp.join(logger, 'videos')

env = make_vec_env('PointMassDense-1-v1', 4,  wrapper_class = FlattenDictWrapper, wrapper_env_kwargs =['observation', 'achieved_goal', 'desired_goal'])
env = VecVideoRecorder(env, video_folder,
                   record_video_trigger=lambda x: x %100000, video_length=400,
                   name_prefix="Video-{}")

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.2) * np.ones(n_actions))

model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)
model.learn(total_timesteps=int(nIter))
model.save(expDir + "/%s/model" %(name, np.format_float_scientific(nIter)))

# del model # remove to demonstrate saving and loading

# model = DDPG.load("ddpg_mountain")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
