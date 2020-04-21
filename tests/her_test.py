from stable_baselines import HER, DQN, SAC, DDPG, TD3, PPO2
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
from stable_baselines.run_utils import record_her_indep
from stable_baselines.common.bit_flipping_env import BitFlippingEnv
import gym, mujoco_py
import os
import os.path as osp
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
import numpy as np
from stable_baselines.common.policies import MlpPolicy, CnnPolicy
import pdb


model_class = DDPG 
goal_selection_strategy = 'future' 
expDir = '/home/shivanik/lab/pointExp/state/' + 'her__0'
verbose = 1
num_objs = 0
logger = osp.join(expDir, 'logs')
video_folder = osp.join(logger, 'videos')
nIter = 1e5

env = gym.make('PointMass-%d-v1' %num_objs)
n_actions = env.action_space.shape[-1]
stddev = 0.2
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.01 * np.ones(n_actions))

policy = 'MlpPolicy'

args_alg = dict(
	random_exploration=0.2,
	buffer_size=int(1E6), 
	batch_size=256,
	nb_eval_steps=10, 
	action_noise=action_noise,
	tensorboard_log=logger,
)


model = HER(policy, env, model_class, n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy, 
                                                verbose=1, **args_alg)
model.learn(int(nIter))
model.save(expDir + "/%s" %np.format_float_scientific(nIter))
#model = HER.load("point1_deter", env=env)

record_her_indep(env, model, expDir, num_files=10, video_len=500)
