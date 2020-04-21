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
import GPUtil
import multiprocessing
from sac_func import *
import pdb


expDir = '/home/shivanik/lab/pointExp/state/'
# expDir = '/Users/samarth/lab/code/pointExp/state/Point0'
num_objs = 1

verbose = 1
name = 'sac_%d_fixed_0.5' %num_objs
nIter = 5e7

save_video_length = 200
save_video_interval = 1000000


def set_gpu():
    gpu = GPUtil.getAvailable(limit=3, excludeID=[0, 1])
    vis_gpu = ""
    for g in gpu:
        vis_gpu += ", " + str(g)
    vis_gpu = vis_gpu[1:]
    os.environ["CUDA_VISIBLE_DEVICES"] = vis_gpu
    print("Setting GPUS: ", vis_gpu)

def func_run(env, logger, lr, action_noise, file):
	expDir = '/home/shivanik/lab/pointExp/state/'
	num_objs = 1

	verbose = 1
	name = 'sac_%d_0.5' %num_objs
	nIter = 5e7

	save_video_length = 200
	save_video_interval = 1000000
	env = VecVideoRecorder(env, osp.join(logger, "videos"), record_video_trigger=lambda x: x % save_video_interval == 0, video_length=save_video_length)
	model = SAC(MlpPolicy, env, 
			verbose=verbose,
            tensorboard_log=logger,
            learning_rate = lr,
            action_noise=action_noise,
            )
	model.learn(total_timesteps=int(nIter), log_interval=100)
	exp_name = expDir + "/%s/%s_%s" %(name, np.format_float_scientific(nIter), np.format_float_scientific(lr))
	model.save(exp_name)
	file.write(exp_name + '\n')
	env.close()
	return True


def train():
	set_gpu()
	expDir = '/home/shivanik/lab/pointExp/state/'
	num_objs = 1

	verbose = 1
	name = 'sac_%d_0.5' %num_objs
	nIter = 1e8

	save_video_length = 200
	save_video_interval = 1000000
	file = open('sac_done.txt', 'w+')
	env = make_vec_env('PointMassDense-%d-v1' %num_objs, 1, wrapper_class = FlattenDictWrapper, wrapper_env_kwargs =['observation', 'achieved_goal', 'desired_goal'])
	n_actions = env.action_space.shape[-1]
	stddev = 0.2


	pool = multiprocessing.Pool(processes=4)
	for lr in [1e-5]: #, 5e-4, 1e-5 
		logger = osp.join(expDir, name, 'logs%s_%s' %(np.format_float_scientific(nIter), np.format_float_scientific(lr)))
		env = VecVideoRecorder(env, osp.join(logger, "videos"), record_video_trigger=lambda x: x % save_video_interval == 0, video_length=save_video_length)
		action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

		# boo = pool.apply_async(func_run, args=(env, logger, lr, action_noise, file))
		model = SAC(MlpPolicy, env, 
					verbose=verbose,
		            tensorboard_log=logger,
		            learning_rate = lr,
		            action_noise=action_noise,
		            )
		model.learn(total_timesteps=int(nIter), log_interval=100)
		exp_name = expDir + "/%s/%s_%s" %(name, np.format_float_scientific(nIter), np.format_float_scientific(lr))
		model.save(exp_name)
		file.write(exp_name + '\n')
		env.close()
	file.close()
	pool.close()
	pool.join()


def play():
	model = SAC.load(expDir + "/%s/%d" %(name,  np.format_float_scientific(nIter)))
	env = gym.make('PointMassDense-1-v1')
	while True:
	    action, _states = model.predict(obs)
	    obs, rewards, dones, info = env.step(action)
	    env.render(mode='human')

def record(exp):
	model = SAC.load(exp)
	env = make_vec_env('PointMassDense-%d-v1' %num_objs, 1, wrapper_class = FlattenDictWrapper, wrapper_env_kwargs =['observation', 'achieved_goal','desired_goal'])
	env = VecVideoRecorder(env, osp.join(logger, "videos_2"), record_video_trigger=lambda x: x % save_video_interval == 0, video_length=save_video_length)
	model.set_env(env)
	model.learn(total_timesteps=2000, log_interval=100)
	# model.save(expDir + "/%s/%d" %(name, nIter))
	env.close()

if __name__ == "__main__":
	train()





