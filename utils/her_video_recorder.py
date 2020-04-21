import gym, mujoco_py
import os, os.path as osp
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
from stable_baselines import HER, DQN, SAC, DDPG, TD3
from gym.wrappers.monitoring.nonVec_video_recorder import NonVecRecorder

def main(env_id):
	env_type = 'robotics'
	env = gym.make(env_id)

	save_file = "/home/shivanik/fetch_trial.zip"
	video_file = "/home/shivanik/fetch_her_videos/"
	env_recorder = NonVecRecorder(env)

	video_len = 100

	model = HER.load(save_file, env=env)
	action_spec = env.action_space
	env.reset()
	i = 0
	record = False
	num_files = 4
	obs = env.reset()
	for i in range(num_files):
		fname = video_file + "%d.mp4" %i
		print(fname)
		env_recorder.init_video_writing(fname=fname)
		for j in range(video_len):
			action, _ = model.predict(obs)
			obs, reward, done, _ = env.step(action)
			env_recorder.viz(True)
			if done:
				obs = env.reset()
		env_recorder.close()
		env.reset()

	env.close()

if __name__ == "__main__":
	env_id = 'FetchReach-v1'
	main(env_id)