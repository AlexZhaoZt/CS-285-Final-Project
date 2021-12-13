import numpy as np
from cs285.infrastructure import pytorch_util as ptu


class PairArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs[0].shape) > 3:
            observation = ptu.from_numpy(obs)
        else:
            observation = ptu.from_numpy(obs)[None]
            # observation = (obs[0][None], obs[1][None])
        ## TODO return the action that maxinmizes the Q-value 
        # at the current observation as the output
        action_vals = ptu.to_numpy(self.critic.forward(observation))
        action = np.argmax(action_vals, -1)
        return action.squeeze()

class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        ## TODO return the action that maxinmizes the Q-value 
        # at the current observation as the output
        action_vals = self.critic.qa_values(observation)
        action = np.argmax(action_vals, -1)
        return action.squeeze()