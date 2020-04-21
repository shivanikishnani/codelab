from stable_baselines import HER, DQN, SAC, DDPG, TD3
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
from stable_baselines.run_utils import record_her_indep
from stable_baselines.common.bit_flipping_env import BitFlippingEnv
import gym, mujoco_py
import os
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
import numpy as np
import pdb

env = gym.make('FetchReach-v1')


model_class = DDPG 
goal_selection_strategy = 'future' 


n_actions = env.action_space.shape[-1]
stddev = 0.2
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.01 * np.ones(n_actions))
policy_kwargs = {}

args_alg = dict(
	random_exploration=0.3,
	buffer_size=int(1E6), 
	batch_size=256,
	nb_eval_steps=10, 
	#actor_lr=1e-3, 
	action_noise=action_noise,
	policy_kwargs = policy_kwargs
)

model = HER('MlpPolicy', env, model_class, n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy,
                                                verbose=1, **args_alg)
# Train the model
model.learn(50000)

model.save("fetch_trial")

# WARNING: you must pass an env
# or wrap your environment with HERGoalEnvWrapper to use the predict method

#model = HER.load('fetch_trial.zip', env=env)
dirs = os.getcwd()
record_her_indep(env, model, dirs)