import tensorflow
import gym
from gym.envs.registration import register
import sys, tty, termios
import readchar

# MACROS
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# Key mapping
arrow_keys = {
    '\x1b[A': UP,
    '\x1b[B': DOWN,
    '\x1b[C': RIGHT,
    '\x1b[D': LEFT
}

# register frozenlake with is_slippery False
register(
    id = 'FrozenLake-v3',
    entry_point = 'gym.envs.toy_text:FrozenLakeEnv',
    kwargs = {'map_name' : '4x4', 'is_slippery' : False}
)

env = gym.make('FrozenLake-v3')
env.render()

while True:
    # choose an action from keyboard
    #key = inkey()
    key = readchar.readkey()
    if key not in arrow_keys.keys():
        print ("Game aborted!")
        break

    action = arrow_keys[key]
    state, reward, done, info = env.step(action)
    env.render() # show the board after action
    print ("State : ", state, " Action: ", action, " Rewared: ", reward, " Info: ", info)

    if done:
        print ("Finished with reward : ", reward)
        break

