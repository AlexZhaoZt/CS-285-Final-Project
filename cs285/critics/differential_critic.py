from .base_critic import BaseCritic
import torch
from torch import nn
from torch import optim
from cs285.infrastructure import pytorch_util as ptu
import itertools

class DifferentialCritic(nn.Module, BaseCritic):
    """
        Notes on notation:

        Prefixes and suffixes:
        ob - observation
        ac - action
        _no - this tensor should have shape (batch self.size /n/, observation dim)
        _na - this tensor should have shape (batch self.size /n/, action dim)
        _n  - this tensor should have shape (batch self.size /n/)

        Note: batch self.size /n/ is defined at runtime.
        is None
    """
    def __init__(self, hparams):
        super().__init__()
        self.ob_dim = hparams['ob_dim']
        self.ac_dim = hparams['ac_dim']
        self.discrete = hparams['discrete']
        self.size = hparams['size']
        self.n_layers = hparams['n_layers']
        self.learning_rate = hparams['learning_rate']

        # critic parameters
        self.num_target_updates = hparams['num_target_updates']
        self.num_grad_steps_per_target_update = hparams['num_grad_steps_per_target_update']
        self.gamma = hparams['gamma']
        # self.critic_network = ptu.build_mlp(
        #     self.ob_dim,
        #     1,
        #     n_layers=self.n_layers,
        #     size=self.size,
        # )
        self.critic_network = ptu.build_mlp(
            2 * self.ob_dim + 1,
            1,
            n_layers=self.n_layers,
            size=self.size,
        )
        self.critic_network.to(ptu.device)
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.critic_network.parameters(),
            self.learning_rate,
        )

    def forward(self, obs1, obs2, terminal):
        return self.critic_network(torch.cat([obs1, obs2, terminal], 1)).squeeze(1)

    def forward_np(self, obs1, obs2, terminal):
        obs1 = ptu.from_numpy(obs1)
        obs2 = ptu.from_numpy(obs2)
        terminal = ptu.from_numpy(terminal).unsqueeze(1)
        predictions = self(obs1, obs2, terminal)
        return ptu.to_numpy(predictions)

    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        """
            Update the parameters of the critic.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                ac_na: length: sum_of_path_lengths. The action taken at the current step.
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                reward_n: length: sum_of_path_lengths. Each element in reward_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end

            returns:
                training loss
        """
        # TODO: Implement the pseudocode below: do the following (
        # self.num_grad_steps_per_target_update * self.num_target_updates)
        # times:
        # every self.num_grad_steps_per_target_update steps (which includes the
        # first step), recompute the target values by
        #     a) calculating V(s') by querying the critic with next_ob_no
        #     b) and computing the target values as r(s, a) + gamma * V(s')
        # every time, update this critic using the observations and targets
        #
        # HINT: don't forget to use terminal_n to cut off the V(s') (ie set it
        #       to 0) when a terminal state is reached
        # HINT: make sure to squeeze the output of the critic_network to ensure
        #       that its dimensions match the reward
        
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na).to(torch.long)
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_n = ptu.from_numpy(reward_n)
        terminal_n = ptu.from_numpy(terminal_n)
        
        # length = ob_no.shape[0]

        # ob_no = ptu.from_numpy(ob_no)[:length-1]
        # ac_na = ptu.from_numpy(ac_na).to(torch.long)[:length-1]
        # next_ob_no = ptu.from_numpy(next_ob_no)[:length-1]
        # reward_n = ptu.from_numpy(reward_n)
        # terminal_n = ptu.from_numpy(terminal_n)[:length-1]
        
        total_loss = 0
        
        # Original

        # for i in range(self.num_target_updates):
        #     V_sp = self.forward(next_ob_no)
        #     target_val = (reward_n + (1 - terminal_n) * self.gamma * V_sp).detach()
        #     for j in range(self.num_grad_steps_per_target_update):
        #         loss = self.loss(target_val, self.forward(ob_no))
        #         total_loss += loss
        #         self.optimizer.zero_grad()
        #         loss.backward()
        #         self.optimizer.step()

        
        # First loss
        idx = torch.nonzero(terminal_n).squeeze() + 1
        split_ob_no = torch.tensor_split(ob_no, idx)
        split_reward_n = torch.tensor_split(reward_n, idx)
        split_next_ob_no = torch.tensor_split(next_ob_no, idx)
        
        reward_cumsums = []

        flag = 0
        ob_no_pairs = []
        terminals = []
        for reward_traj, ob_traj, next_ob_traj in zip(split_reward_n, split_ob_no, split_next_ob_no):
            # indices = np.arange(reward_traj.shape[0])
            # id_all_pairs = itertools.combinations(indices, r=2)
            exps = torch.arange(reward_traj.shape[0])
            # print(ob_traj.shape)
            gamma_exps = self.gamma ** exps
            if ob_traj.shape[0]:
                ob_traj = torch.cat((ob_traj, next_ob_traj[-1].unsqueeze(0)), 0)
            # print(ob_traj.shape)
            for i in range(reward_traj.shape[0]):
                # reward_cumsum_i_to_end = torch.cumsum(gamma_exps[:reward_traj.shape[0] - 1 - i] * reward_traj[i:-1], 0)
                reward_cumsum_i_to_end = torch.cumsum(gamma_exps[:reward_traj.shape[0] - i] * reward_traj[i:], 0)
                # reward_cumsum_i_to_end = torch.cumsum(gamma_exps[i:] * reward_traj[i:], 0)
                j_list = range(i+1, ob_traj.shape[0])
                # j_list = range(i+1, reward_traj.shape[0])
                flag += 1
                ob_no_pairs_i_to_end = torch.stack((ob_traj[i].expand(len(j_list), -1), ob_traj[j_list]), 1)
                # print(ob_no_pairs_i_to_end.shape)
                # print(terminal_i_to_end)
                # assert(flag != 3)
                # non_terminal_i_to_end = torch.zeros(ob_no_pairs_i_to_end.shape[0])
                terminal_i_to_end = torch.zeros(ob_no_pairs_i_to_end.shape[0])
                if terminal_i_to_end.shape[0]:
                    terminal_i_to_end[-1] = 1
                # terminals.append(non_terminal_i_to_end)
                terminals.append(terminal_i_to_end)
                # ob_no_pairs.append(ob_no_pairs_i_to_end)
                ob_no_pairs.append(ob_no_pairs_i_to_end)
                reward_cumsums.append(reward_cumsum_i_to_end)
                # reward_cumsums.append(reward_traj[i:])
        reward_cumsums = torch.cat(reward_cumsums)
        ob_no_pairs = torch.cat(ob_no_pairs)
        terminals = torch.cat(terminals).unsqueeze(1)
        ob_no_1s = ob_no_pairs[:,0]
        ob_no_2s = ob_no_pairs[:,1]
        for i in range(self.num_target_updates):
            for j in range(self.num_grad_steps_per_target_update):
                G_sp = self.forward(ob_no_1s, ob_no_2s, terminals)
                loss = self.loss(G_sp, reward_cumsums)
                total_loss += loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()




        # # Third loss
        # reward_diff = self.gamma * reward_n[1:length] - reward_n[:length-1] 
        # for i in range(self.num_target_updates):
        #     for j in range(self.num_grad_steps_per_target_update):
        #         G_sp = self.forward(next_ob_no, ob_no)
        #         loss = self.loss((1 - terminal_n) * reward_diff, (1 - terminal_n) * G_sp)
        #         total_loss += loss
        #         self.optimizer.zero_grad()
        #         loss.backward()
        #         self.optimizer.step()

        return total_loss
