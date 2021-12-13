import numpy as np
import torch
from cs285.infrastructure.dqn_utils import PairReplayBuffer, PiecewiseSchedule
from cs285.policies.argmax_policy import ArgMaxPolicy, PairArgMaxPolicy
from cs285.critics.dqn_critic import DQNCritic
from cs285.infrastructure import pytorch_util as ptu
from torch.nn import utils
from torch import nn
import gym
import copy

class PairEnv(object):
    def __init__(self, env_id):
        self.env_id = env_id
        self.env1 = gym.make(env_id)
        # self.env2 = copy.deepcopy(self.env1)
        self.env2 = gym.make(env_id)
        # self.env2 = copy.deepcopy(self.env)

        self.observation_space = self.env1.observation_space
        self.action_space = self.env1.action_space
        self.reward = 0

    def step(self, action):
        o1, r1, d1, info = self.env1.step(action[0])
        o2, r2, d2, info = self.env2.step(action[1])
        self.alive_time = 1
        self.reward += r1
        pair_reward = self.modifiedReward(r1, d1) - self.modifiedReward(r2, d2)
        return ([o1, o2], pair_reward, d1 or d2, info)

    def modifiedReward(self, r, d):
        if self.env_id == "CartPole-v0":
            return r  - self.alive_time * (d == True)
        elif self.env_id == "MountainCar-v0":
            return r + self.alive_time * (d == True)
        else:
            return r

    def reset(self):
        self.reward = 0
        self.alive_time = 0
        return [self.env1.reset(), self.env2.reset()]

    def close(self):
        self.env1.close()
        self.env2.close()

    def seed(self, seed):
        self.env1.seed(seed)
        self.env2.seed(seed)


class DDQNCritic:
    def __init__(self, hparams, critic, action_space):
        self.model = critic
        self.action_space = action_space
        self.double_q = hparams['double_q']
        self.gamma = hparams['gamma']
        self.loss = nn.SmoothL1Loss()  # AKA Huber loss
        self.grad_norm_clipping = hparams['grad_norm_clipping']


    def forward(self, obs):
        obs1 = obs[:,0]
        obs2 = obs[:,1]
        # obs1 = ptu.from_numpy(obs1)
        # obs2 = ptu.from_numpy(obs2)

        out1 = self.model.q_net(obs1)
        out2 = self.model.q_net(obs2)
        out1_extended = torch.repeat_interleave(out1, self.action_space, 1)
        out2_extended = torch.tile(out2, (1, self.action_space))
        diff = out1_extended - out2_extended
        return diff
        return ptu.to_numpy(diff)


    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na).to(torch.long)
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_n = ptu.from_numpy(reward_n)
        terminal_n = ptu.from_numpy(terminal_n)
        
        # compute the Q-values from the target network 
        qa_t_values = self.forward(ob_no)
        action_index = ac_na[:,0] * self.action_space + ac_na[:,1]
        q_t_values = torch.gather(qa_t_values, 1, action_index.unsqueeze(1)).squeeze(1)

        if self.double_q:
            # You must fill this part for Q2 of the Q-learning portion of the homework.
            # In double Q-learning, the best action is selected using the Q-network that
            # is being updated, but the Q-value for this action is obtained from the
            # target Q-network. Please review Lecture 8 for more details,
            # and page 4 of https://arxiv.org/pdf/1509.06461.pdf is also a good reference.
            v1s = self.model.q_net(next_ob_no[:,0])
            v2s = self.model.q_net(next_ob_no[:,1])
            a_tp1_1 = v1s.argmax(dim=1)
            a_tp1_2 = v2s.argmax(dim=1)

            v1s_target = self.model.q_net_target(next_ob_no[:,0])
            v2s_target = self.model.q_net_target(next_ob_no[:,1])

            q_tp1 = torch.gather(v1s_target, 1, a_tp1_1.unsqueeze(1)).squeeze(1) - torch.gather(v2s_target, 1, a_tp1_2.unsqueeze(1)).squeeze(1)
            # q_tp1, _ = qa_tp1_values.max(dim=1)


            
        else:
            v1, _ = torch.max(self.model.q_net_target(next_ob_no[:,0]), 1)
            v2, _ = torch.max(self.model.q_net_target(next_ob_no[:,1]), 1)
            q_tp1 = v1 - v2
            
        # TODO compute targets for minimizing Bellman error
        # HINT: as you saw in lecture, this would be:
            #currentReward + self.gamma * qValuesOfNextTimestep * (not terminal)
        target = (reward_n + self.gamma * q_tp1 * (1-terminal_n))
        target = target.detach()

        assert q_t_values.shape == target.shape
        loss = self.loss(q_t_values, target)

        self.model.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self.model.q_net.parameters(), self.grad_norm_clipping)
        self.model.optimizer.step()
        self.model.learning_rate_scheduler.step()
        return {
            'Training Loss': ptu.to_numpy(loss),
        }


class DDQNAgent(object):
    def __init__(self, env, agent_params):
        
        self.env_id = agent_params['env_name']
        self.env = PairEnv(self.env_id) #TODO make pairEnv
        self.env.seed(agent_params['seed'])
        self.agent_params = agent_params
        self.batch_size = agent_params['batch_size']
        # import ipdb; ipdb.set_trace()
        self.last_obs = self.env.reset()

        self.num_actions = agent_params['ac_dim']
        self.learning_starts = agent_params['learning_starts']
        self.learning_freq = agent_params['learning_freq']
        self.target_update_freq = agent_params['target_update_freq']

        self.replay_buffer_idx = None
        self.exploration = agent_params['exploration_schedule']
        self.optimizer_spec = agent_params['optimizer_spec']

        self.critic = DQNCritic(agent_params, self.optimizer_spec)
        self.pair_critic = DDQNCritic(agent_params, self.critic, self.num_actions)
        self.actor = PairArgMaxPolicy(self.pair_critic)
        self.eval_actor = ArgMaxPolicy(self.critic)
        lander = agent_params['env_name'].startswith('LunarLander')
        self.replay_buffer = PairReplayBuffer(
            agent_params['replay_buffer_size'], agent_params['frame_history_len'], lander=lander)
        self.t = 0
        self.num_param_updates = 0

    def add_to_replay_buffer(self, paths):
        pass

    def step_env(self):
        """
            Step the env and store the transition
            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.
            Note that self.last_obs must always point to the new latest observation.
        """        

        # TODO store the latest observation ("frame") into the replay buffer
        # HINT: the replay buffer used here is `PairReplayBuffer`
            # in dqn_utils.py
        self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)

        eps = self.exploration.value(self.t)

        # TODO use epsilon greedy exploration when selecting action
        perform_random_action = np.random.rand() < eps or self.t < self.learning_starts
        if perform_random_action:
            # HINT: take random action 
                # with probability eps (see np.random.random())
                # OR if your current step number (see self.t) is less that self.learning_starts
            action1 = np.random.randint(self.num_actions)
        else:
            # HINT: Your actor will take in multiple previous observations ("frames") in order
                # to deal with the partial observability of the environment. Get the most recent 
                # `frame_history_len` observations using functionality from the replay buffer,
                # and then use those observations as input to your actor. 
            frame_history = self.replay_buffer.encode_recent_observation()
            
            # merged_action_index = self.actor.get_action(frame_history) 
            # action1 = merged_action_index // self.num_actions
            # action2 = merged_action_index % self.num_actions

            action1 = self.eval_actor.get_action(frame_history[0]) 
        perform_random_action = np.random.rand() < eps or self.t < self.learning_starts
        if perform_random_action:
            # HINT: take random action 
                # with probability eps (see np.random.random())
                # OR if your current step number (see self.t) is less that self.learning_starts
            action2 = np.random.randint(self.num_actions)
        else:
            # HINT: Your actor will take in multiple previous observations ("frames") in order
                # to deal with the partial observability of the environment. Get the most recent 
                # `frame_history_len` observations using functionality from the replay buffer,
                # and then use those observations as input to your actor. 
            frame_history = self.replay_buffer.encode_recent_observation()
            
            # merged_action_index = self.actor.get_action(frame_history) 
            # action1 = merged_action_index // self.num_actions
            # action2 = merged_action_index % self.num_actions

            action2 = self.eval_actor.get_action(frame_history[1]) 
        
        action = (action1, action2)
        # TODO take a step in the environment using the action from the policy
        # HINT1: remember that self.last_obs must always point to the newest/latest observation
        # HINT2: remember the following useful function that you've seen before:
            #obs, reward, done, info = env.step(action)
        self.last_obs, reward, done, info = self.env.step(action)

        # TODO store the result of taking this action into the replay buffer
        # HINT1: see your replay buffer's `store_effect` function
        # HINT2: one of the arguments you'll need to pass in is self.replay_buffer_idx from above
        self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)

        # TODO if taking this step resulted in done, reset the env (and the latest observation)
        if done:
            self.last_obs = self.env.reset()

    def sample(self, batch_size):
        if self.replay_buffer.can_sample(self.batch_size):
            return self.replay_buffer.sample(batch_size)
        else:
            return [],[],[],[],[]

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        log = {}
        if (self.t > self.learning_starts
                and self.t % self.learning_freq == 0
                and self.replay_buffer.can_sample(self.batch_size)
        ):

            # TODO fill in the call to the update function using the appropriate tensors
            log = self.pair_critic.update(
                ob_no, ac_na, next_ob_no, re_n, terminal_n   
            )
            # TODO update the target network periodically 
            # HINT: your critic already has this functionality implemented
            if self.num_param_updates % self.target_update_freq == 0:
                self.critic.update_target_network()

            self.num_param_updates += 1

        self.t += 1
        return log
