import gym
import mujoco_py
import random
from PIL import Image
import numpy as np
import six
import os
import ast
import pdb

def test():
	file = os.getcwd() + "/Pictures2/coords.txt"
	# d = [ast.literal_eval(line.rstrip("\n")) for line in file]
	# print(d)
	d = []
	with open(file , 'r') as f:
		lines = f.readlines()
		for i in range(len(lines)):
			line = lines[i]
			d.append(ast.literal_eval(line.rstrip("\n")))
	print(d)

def make_dir(folder, typeE):
	if not os.path.exists(folder):
		print('Making directory: {}'.format(folder))
		os.makedirs(folder)
		return folder
	else:
		print('Existing directory: {}'.format(folder))
		folder_name = folder.split('/')[-1]
		count = sum([folder_name in name for name in os.listdir('/home/shivanik/lab/' + typeE)])
		folder = folder + "_" + str(count + 1)
		return make_dir(folder, typeE)

def test2():
	dirs = ['ag']
	for j, fol in enumerate(dirs):
		env = gym.make("PointMassDense-%d-v1" % (j + 1))
		file = open(os.getcwd() + "/%s/coords.txt" %fol,"w+")
		env.reset()
		num = 5
		goal = []
		
		for i in range(num):
			print("%d", i, end="\r")
			pos = np.random.uniform(-0.3, 0.3, size=2)
			env.set_agent0_xy(pos)
			for _ in range(2):
				env.step(env.action_space.sample())

			agent_pos = env.sim.data.body_xpos[env.agent_id][0:2]

			data = env.render(mode='rgb_array', height=224, width=224)
			data = np.asarray(data, dtype=np.uint8)
			im = Image.fromarray(data)
			name = '%s/train%d.png' %(fol,i)
			im.save(name)

			goal = str(np.concatenate([agent_pos.copy(), env.unwrapped.ret_obs()]).tolist())+"\n"
			env.reset()
			file.write(goal)

			env.reset()

		env.close()
		file.close()


def generate_few():
	num_objects = [2]
	fol = '.'
	for j in [2]:
		env = gym.make("PointMassDense-%d-v1" % (j))
		env.reset()
		num = 5
		goal = []
		
		for i in range(num):
			print("%d", i, end="\r")

			data = env.render(mode='rgb_array', height=84, width=84)
			data = np.asarray(data, dtype=np.uint8)
			im = Image.fromarray(data)
			name = '%s/train%d.png' %(fol,i)
			im.save(name)
			env.reset()

		env.close()

if __name__ == '__main__':
	generate_few()
