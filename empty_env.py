import numpy as np
import random
class Environment:
    def __init__(self):
        self.observation_space = np.array([random.random() for i in range(4)])
        self.action_space = Action_space(2)

    def reset(self):
        return np.array([0., 0., 0., 0.])
    def step(self,action):
        obs_next = np.array([random.random() for i in range(4)])
        reward = 1.0
        if random.randint(0,10) >5:
            done = True
        else:
            done = False
        _ = {}
        return obs_next, reward, done, _

class Action_space:
    def __init__(self,n):
        self.n = n

