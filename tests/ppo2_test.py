import gym
import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines import PPO2, HER
import os.path as osp
import time
from stable_baselines.common.vec_env import VecVideoRecorder, DummyVecEnv
from gym.wrappers import FlattenDictWrapper

expDir = '/home/shivanik/lab/pointExp/state/'
verbose = 1
num_objs = 1
name = 'ppo2_%d' %num_objs
logger = osp.join(expDir, name, 'logs')
video_folder = osp.join(logger, 'videos')
nIter = 1e8
save_video_interval = 1000000
save_video_length = 200

# multiprocess environment
def learn():
	# expDir = '/home/shivanik/lab/pointExp/state/'
	# verbose = 1
	# num_objs = 1
	# name = 'ppo2_%d' %num_objs
	# logger = osp.join(expDir, name, 'logs')
	# video_folder = osp.join(logger, 'videos')
	# nIter = 1e7
	# save_video_interval = 5000

	env = make_vec_env('PointMassDense-%d-v1' %num_objs, 1,  wrapper_class = FlattenDictWrapper, wrapper_env_kwargs =['observation', 'achieved_goal', 'desired_goal'])
	env = VecVideoRecorder(env, video_folder,
                       record_video_trigger=lambda x: x % save_video_interval == 0, video_length=save_video_length,
                       name_prefix="Video-{}")

	model = PPO2(MlpPolicy, env, verbose=verbose,
	            tensorboard_log=logger,)
	model.learn(total_timesteps=int(nIter))
	model.save(expDir + "/%s/%s" %(name, np.format_float_scientific(nIter)))

	# del model # remove to demonstrate saving and loading
	# model = PPO2.load("ppo2_cartpole")
	# obs = env.reset()
	# record_her_indep(env, model, file = logger)
	# env.close()

def _load(model_name):
	model = PPO2.load(model_name)
	env = make_vec_env('PointMassDense-%d-v1' %num_objs, 1, wrapper_class = FlattenDictWrapper, wrapper_env_kwargs =['observation', 'achieved_goal', 'desired_goal'])
	env = VecVideoRecorder(env, osp.join(logger, "videos_3"), record_video_trigger=lambda x: x % save_video_interval == 0, video_length=save_video_length)
	model.set_env(env)
	model.learn(total_timesteps=int(nIter), log_interval=100)
	# model.save(exp_name)
	model.save(model_name + "_new")
	env.close()

def record_her_indep(env, model, file, num_files=5, video_len=400):
	logger = file
	env_recorder = NonVecRecorder(env)

	video_file = osp.join(logger, "videos")
	if not os.path.exists(video_file):
		os.makedirs(video_file)
	
	for i in range(num_files):
		fname = video_file + "/%d.mp4" %i
		print(fname)
		env_recorder.init_video_writing(fname=fname)
		obs = env.reset()
		for j in range(video_len):
			action, _ = model.predict(obs)
			obs, reward, done, _ = env.step(action)
			env_recorder.viz(True)
			if done:
				obs = env.reset()
		env_recorder.close()
		env.reset()

def play():
	# model = PPO2('Users/samarth/lab/pointExp/state/ppo2_0')
	env = gym.make('PointMassDense-0-v1')
	env.reset()

	while True:
		time.sleep(2)
		env.reset()
		env.render()


# _load("/home/shivanik/lab/pointExp/state/ppo2_1/mod")
learn()
