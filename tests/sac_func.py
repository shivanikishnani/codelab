from stable_baselines.common.vec_env import VecVideoRecorder

from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec

import gym
import os
from gym.wrappers import FlattenDictWrapper
import numpy as np
from stable_baselines.common.cmd_util import make_vec_env
import os.path as osp
import pdb

expDir = '/home/shivanik/lab/pointExp/state/'
# expDir = '/Users/samarth/lab/code/pointExp/state/Point0'
num_objs = 1

verbose = 1
name = 'sac_%d_fixed_0.5' %num_objs
nIter = 1e7

save_video_length = 200
save_video_interval = 5000

def run_experiment(verbose, tensorboard_log, learning_rate):
	pdb.set_trace()
	env = make_vec_env('PointMassDense-%d-v1' %num_objs, 1, wrapper_class = FlattenDictWrapper, wrapper_env_kwargs =['observation', 'achieved_goal', 'desired_goal'])
	env = VecVideoRecorder(env, osp.join(logger, "videos"), record_video_trigger=lambda x: x % save_video_interval == 0, video_length=save_video_length)

	n_actions = env.action_space.shape[-1]
	stddev = 0.2
	action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

	model = SAC(MlpPolicy, env, 
				verbose=verbose,
	            tensorboard_log=logger,
	            learning_rate = learning_rate,
	            action_noise=action_noise,
	            )
	model.learn(total_timesteps=int(nIter), log_interval=100)
	model.save(expDir + "/%s/%s_%s" %(name, np.format_float_scientific(nIter), np.format_float_scientific(learning_rate)))
	env.close()