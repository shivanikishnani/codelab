import gym, mujoco_py
import keyboard

env = gym.make('PointMassDense-2-v1')
action_spec = env.action_space()
env.reset()
while True:
        # for _ in range(1000):
    action = action_spec.sample()
    action[:2] = 0.
    if keyboard.is_pressed('l'):
      action[1] = action_spec.high[1]
    elif keyboard.is_pressed('.'):
      action[1] = action_spec.low[1]
    elif keyboard.is_pressed(','):
      action[0] = action_spec.low[0]
    elif keyboard.is_pressed('/'):
      action[0] = action_spec.high[0]
    action *= args.action_multiplier
    env.step(action)