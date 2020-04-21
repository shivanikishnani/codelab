import gym, mujoco_py 
import time
import pdb

env = gym.make('PointMassDense-2-v1')
print(env.reset())
print(env.num_objs)
# pdb.set_trace()
for i in range(200):
	env.reset()
	env.step(env.action_space.sample())
	time.sleep(1)
	# pdb.set_trace()
	kwargs={'width':84, 'height':84}
	env.render('human', **kwargs)
