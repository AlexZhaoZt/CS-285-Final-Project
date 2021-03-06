from collections import OrderedDict
import pickle
import os
import sys
import time

import gym
from gym import wrappers
import numpy as np
from cs285.agents.dac_agent import DACAgent
import torch
from cs285.infrastructure import pytorch_util as ptu

from cs285.infrastructure.logger import Logger

from cs285.agents.dqn_agent import DQNAgent
from cs285.agents.ddqn_agent import DDQNAgent
from cs285.infrastructure.dqn_utils import (
        get_wrapper_by_name,
        register_custom_envs,
)
from gym_minigrid.wrappers import *
from cs285.infrastructure import utils
import copy
# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40 # we overwrite this in the code below


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


class RL_Trainer(object):

    def __init__(self, params):

        #############
        ## INIT
        #############

        # Get params, create logger
        self.params = params
        self.logger = Logger(self.params['logdir'])

        # Set random seeds
        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(
            use_gpu=not self.params['no_gpu'],
            gpu_id=self.params['which_gpu']
        )

        #############
        ## ENV
        #############

        # Make the gym environment
        register_custom_envs()
        if self.params['agent_class'] == DDQNAgent or self.params['agent_class'] == DACAgent: 
            self.env = PairEnv(self.params['env_name'])
        else:
            self.env = gym.make(self.params['env_name'])
        self.eval_env = gym.make(self.params['env_name'])
        # self.env = gym.make(self.params['env_name'])
        # self.env = FullyObsWrapper(self.env)
        # self.env = FlatObsWrapperNoMission(self.env)
        # print(self.env.observation_space)
        if 'env_wrappers' in self.params:
            # These operations are currently only for Atari envs
            self.eval_env = wrappers.Monitor(
                self.eval_env,
                os.path.join(self.params['logdir'], "gym"),
                force=True,
                video_callable=(None if self.params['video_log_freq'] > 0 else False),
            )
            # self.env = params['env_wrappers'](self.env)
            self.eval_env = params['env_wrappers'](self.eval_env)
            self.mean_episode_reward = -float('nan')
            self.best_mean_episode_reward = -float('inf')
        # if 'non_atari_colab_env' in self.params and self.params['video_log_freq'] > 0:
        #     self.env = wrappers.Monitor(
        #         self.env,
        #         os.path.join(self.params['logdir'], "gym"),
        #         force=True,
        #         video_callable=(None if self.params['video_log_freq'] > 0 else False),
        #     )
        #     self.mean_episode_reward = -float('nan')
        #     self.best_mean_episode_reward = -float('inf')

        self.env.seed(seed)
        self.eval_env.seed(seed)

        # import plotting (locally if 'obstacles' env)
        if not(self.params['env_name']=='obstacles-cs285-v0'):
            import matplotlib
            matplotlib.use('Agg')

        # Maximum length for episodes
        self.params['ep_len'] = self.params['ep_len'] or self.env.spec.max_episode_steps
        global MAX_VIDEO_LEN
        MAX_VIDEO_LEN = self.params['ep_len']

        # Is this env continuous, or self.discrete?
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        # Are the observations images?
        print(self.env.observation_space)
        img = len(self.env.observation_space.shape) > 2

        self.params['agent_params']['discrete'] = discrete

        # Observation and action sizes

        ob_dim = self.env.observation_space.shape if img else self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        self.params['agent_params']['ac_dim'] = ac_dim
        self.params['agent_params']['ob_dim'] = ob_dim

        # simulation timestep, will be used for video saving
        if 'model' in dir(self.env):
            self.fps = 1/self.env.model.opt.timestep
        elif 'env_wrappers' in self.params:
            self.fps = 30 # This is not actually used when using the Monitor wrapper
        elif 'video.frames_per_second' in self.eval_env.env.metadata.keys():
            self.fps = self.eval_env.env.metadata['video.frames_per_second']
        else:
            self.fps = 10


        #############
        ## AGENT
        #############
    
        agent_class = self.params['agent_class']
        self.agent = agent_class(self.env, self.params['agent_params'])

    def run_training_loop(self, n_iter, collect_policy, eval_policy,
                          initial_expertdata=None, relabel_with_expert=False,
                          start_relabel_with_expert=1, expert_policy=None):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expertdata:
        :param relabel_with_expert:  whether to perform dagger
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        :param expert_policy:
        """

        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        print_period = 1000 if isinstance(self.agent, DQNAgent) or isinstance(self.agent, DDQNAgent) else 1
        for itr in range(n_iter):
            if itr % print_period == 0:
                print("\n\n********** Iteration %i ************"%itr)

            # decide if videos should be rendered/logged at this iteration
            if itr % self.params['video_log_freq'] == 0 and self.params['video_log_freq'] != -1:
                self.logvideo = True
            else:
                self.logvideo = False

            # decide if metrics should be logged
            if self.params['scalar_log_freq'] == -1:
                self.logmetrics = False
            elif itr % self.params['scalar_log_freq'] == 0:
                self.logmetrics = True
            else:
                self.logmetrics = False

            # collect trajectories, to be used for training
            if isinstance(self.agent, DQNAgent) or isinstance(self.agent, DDQNAgent):
                # only perform an env step and add to replay buffer for DQN
                self.agent.step_env()
                envsteps_this_batch = 1
                train_video_paths = None
                paths = None
            else:
                use_batchsize = self.params['batch_size']
                if itr==0:
                    use_batchsize = self.params['batch_size_initial']
                paths, envsteps_this_batch, train_video_paths = (
                    self.collect_training_trajectories(
                        itr, initial_expertdata, collect_policy, use_batchsize)
                )

            self.total_envsteps += envsteps_this_batch

            # add collected data to replay buffer 
            self.agent.add_to_replay_buffer(paths)

            # train agent (using sampled data from replay buffer)
            if itr % print_period == 0:
                print("\nTraining agent...")
            all_logs = self.train_agent()

            # log/save
            if self.logvideo or self.logmetrics:
                # perform logging
                print('\nBeginning logging procedure...')
                if isinstance(self.agent, DQNAgent) or isinstance(self.agent, DDQNAgent):
                    self.perform_dqn_logging(all_logs)
                else:
                    self.perform_logging(itr, paths, eval_policy, train_video_paths, all_logs)

                if self.params['save_params']:
                    self.agent.save('{}/agent_itr_{}.pt'.format(self.params['logdir'], itr))

    ####################################
    ####################################

    def collect_training_trajectories(self, itr, initial_expertdata, collect_policy, num_transitions_to_sample, save_expert_data_to_disk=False):
        """
        :param itr:
        :param load_initial_expertdata:  path to expert data pkl file
        :param collect_policy:  the current policy using which we collect data
        :param num_transitions_to_sample:  the number of transitions we collect
        :return:
            paths: a list trajectories
            envsteps_this_batch: the sum over the numbers of environment steps in paths
            train_video_paths: paths which also contain videos for visualization purposes
        """
        # TODO: get this from hw1 or hw2
        if itr == 0:
            num_transitions_to_sample = self.params['batch_size_initial']
        else:
            num_transitions_to_sample = self.params['batch_size']

        print("\nCollecting data to be used for training...")
        paths, envsteps_this_batch = utils.sample_trajectories(
            self.env, collect_policy, num_transitions_to_sample, self.params['ep_len'], DAC=isinstance(self.env, PairEnv))

        train_video_paths = self.params['logdir']
        
        return paths, envsteps_this_batch, train_video_paths


    def train_agent(self):
        all_logs = []
        for train_step in range(self.params['num_agent_train_steps_per_iter']):
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(self.params['train_batch_size'])
            train_log = self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
            all_logs.append(train_log)
        return all_logs

    ####################################
    ####################################
    def perform_dqn_logging(self, all_logs):
        
        last_log = all_logs[-1]

        self.run_eval_loop(100)

        episode_rewards = get_wrapper_by_name(self.eval_env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            self.mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            self.best_mean_episode_reward = max(self.best_mean_episode_reward, self.mean_episode_reward)

        logs = OrderedDict()

        logs["Train_EnvstepsSoFar"] = self.agent.t
        print("Timestep %d" % (self.agent.t,))
        if self.mean_episode_reward > -5000:
            logs["Train_AverageReturn"] = np.mean(self.mean_episode_reward)
        print("mean reward (100 episodes) %f" % self.mean_episode_reward)
        if self.best_mean_episode_reward > -5000:
            logs["Train_BestReturn"] = np.mean(self.best_mean_episode_reward)
        print("best mean reward %f" % self.best_mean_episode_reward)

        if self.start_time is not None:
            time_since_start = (time.time() - self.start_time)
            print("running time %f" % time_since_start)
            logs["TimeSinceStart"] = time_since_start

        logs.update(last_log)

        sys.stdout.flush()

        for key, value in logs.items():
            print('{} : {}'.format(key, value))
            self.logger.log_scalar(value, key, self.agent.t)
        print('Done logging...\n\n')

        self.logger.flush()


    def run_eval_loop(self, n_iter):
        """
        :param n_iter:  number of evaluation iterations
        """

        # init vars at beginning of training

        for _ in range(n_iter):
            state = self.eval_env.reset()
            done = False
            while not done:
                # for itr in range(self.params['ep_len']):
                action = self.agent.eval_actor.get_action(state)
                
                next_state, reward, done, info = self.eval_env.step(action)

                state = next_state

        print('-------Done evaluation-------')                    

    def perform_logging(self, itr, paths, eval_policy, train_video_paths, all_logs):

        last_log = all_logs[-1]

        # self.run_eval_loop(self.params['eval_batch_size'])
        #######################

        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(self.eval_env, eval_policy, self.params['eval_batch_size'], self.params['ep_len'])

        # save eval rollouts as videos in tensorboard event file
        # if self.logvideo and train_video_paths != None:
        if self.logvideo:
            print('\nCollecting video rollouts eval')
            eval_video_paths = utils.sample_n_trajectories(self.eval_env, eval_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

            #save train/eval videos
            print('\nSaving train rollouts as videos...')
            # self.logger.log_paths_as_videos(train_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
            #                                 video_title='train_rollouts')
            self.logger.log_paths_as_videos(eval_video_paths, itr, fps=self.fps,max_videos_to_save=MAX_NVIDEO,
                                             video_title='eval_rollouts')

        #######################

        # save eval metrics
        if self.logmetrics:
            # returns, for logging
            train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

            # episode lengths, for logging
            train_ep_lens = [len(path["reward"]) for path in paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

            # decide what to log
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)
            logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time
            logs.update(last_log)

            if itr == 0:
                self.initial_return = np.mean(train_returns)
            logs["Initial_DataCollection_AverageReturn"] = self.initial_return

            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')

            self.logger.flush()