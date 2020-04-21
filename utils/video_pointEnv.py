import gym, mujoco_py
#from stable_baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
#from stable_baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
#from stable_baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
import os, os.path as osp
from 

# from pynput.keyboard import Key, Controller, Listener

# keyboard = Controller()
env_id = 'PointMassDense-2-v1'
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





while i < 1000:
	i += 1
	action = action_spec.sample()
	print(action)
	env.step(action)
	data = env.render('rgb_array')
	
env.close()
	# # Collect events until released
	# with Listener(
	# 		on_press=on_press) as listener:
	# 	try:
	# 		env.step(action)
	# 		env.render()
	# 		listener.join()
	# 	except MyException as e:
	# 		print('{0} was pressed'.format(e.args[0]))
	  
	 