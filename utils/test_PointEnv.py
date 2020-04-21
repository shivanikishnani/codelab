import gym, mujoco_py
#from stable_baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
#from stable_baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
#from stable_baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
import os, os.path as osp
import time

env_id = 'PointMass-1-v1'
env_type = 'robotics'
env = gym.make(env_id)

flatten_dict_observations = True

#env = make_vec_env(env_id, env_type, 1, None, reward_scale=1, flatten_dict_observations=flatten_dict_observations)
#save_video_interval = 1000
#save_video_length = 1000
#env = env = VecVideoRecorder(env, osp.join(os.getcwd(), "videos_smaller"), record_video_trigger=lambda x: x % save_video_interval == 0, video_length=save_video_length)
action_spec = env.action_space
env.reset()
i = 0

while True:
	i += 1
	action = action_spec.sample()
	env.step(action)
	data = env.render()
	# time.sleep(2)
	# env.reset()
	
env.close()
