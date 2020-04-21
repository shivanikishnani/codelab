import gym
import numpy as np

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env
from stable_baselines import DDPG
import os.path as osp
import os
from stable_baselines.common.vec_env.vec_video_recorder import VecVideoRecorder

env_id = 'FetchReachDense-v1'
env_type = 'robotics'
nenv = 1
seed = 1
flatten_dict_observations = True
env = make_vec_env(env_id, env_type, nenv, seed, flatten_dict_observations=flatten_dict_observations)
save_video_interval = 0
save_video_length = 1000
#env = VecVideoRecorder(env, osp.join(os.getcwd(), "stable_baselineDDPG"), record_video_trigger=lambda x: x % save_video_interval == 0, video_length=save_video_length)
# env = DummyVecEnv([lambda: env])

n_actions = env.action_space.shape[-1]
param_noise = None
stddev = 0.2
param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev), adoption_coefficient=1.01)
action_noise = None
logger = "/home/shivanik/pointExp/DDPGComparison/stable_baselines/ddpgParamNoise/log"

model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, tensorboard_log=logger, action_noise=action_noise)
model.learn(total_timesteps=400)
model.save("/home/shivanik/pointExp/DDPGComparison/stable_baselines/ddpgParamNoise")

env.close()
# del model # remove to demonstrate saving and loading

# model = DDPG.load("ddpg_fetch")

# obs = env.reset()

# for i in range(10000):
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     # env.render(mode='rgb_array')
