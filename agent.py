import torch
import random, numpy as np
from pathlib import Path

from neural import DDQNet
from collections import deque

class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.use_cuda = torch.cuda.is_available()
        self.net = DDQNet(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device="cuda")

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.save_every = 5e5
        self.memory = deque(maxlen=7000)
        self.batch_size = 32

        self.gamma = 0.9
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.burnin = 1e4 # min no of exp before training
        self.learn_every = 3 # no of exp between updates to Q_online
        self.sync_every = 1e4 # no of exp between Q_target and Q_online sync

    def act(self, state):
        """
        When the agent is in a state, it chooses an action based on epsilon greedy
        Inputs: state = A single obs of the current state, dim is state_dim
        Outputs: action_idx(int) = An integer representing which action Mario will perform
        """
        #Explore
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        #Exploit
        else:
            state = state.__array__()
            if self.use_cuda:
                state = torch.tensor(state, dtype=torch.float32).cuda()
            else:
                state = torch.tensor(state, dtype=torch.float32)
            state = state.unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease the exploration rate
        self.exploration_rate *= self.exploration_rate_decay
        self.explortaion_rate = max(self.exploration_rate_min, self.exploration_rate)

        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """Add the exp to the memory"""
        state = state.__array__()
        next_state = next_state.__array__()

        if self.use_cuda:
            state = torch.tensor(state, dtype=torch.float32).cuda()
            next_state = torch.tensor(next_state, dtype=torch.float32).cuda()
            action = torch.tensor([action], dtype=torch.float32).cuda()
            reward = torch.tensor([reward], dtype=torch.float32).cuda()
            done = torch.tensor([done], dtype=torch.float32).cuda()
        else:
            state = torch.tensor(state, dtype=torch.float32)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            action = torch.tensor([action], dtype=torch.float32)
            reward = torch.tensor([reward], dtype=torch.float32)
            done = torch.tensor([done], dtype=torch.float32)

        self.memory.append((state, next_state, action, reward, done,))

    def recall(self):
        """Sample exp from memory"""
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):

        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action.long()
        ]
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action.long()
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())


    def learn(self):
        """Update online action value (Q) func with a batch of agent exp"""
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Take a sample from the memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # BackP loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)

    def save(self):
        save_path = (self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt")
        torch.save(dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate), save_path,)
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=('cuda' if self.use_cuda else 'cpu'))
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')

        print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
        self.net.load_state_dict(state_dict)
        self.exploration_rate = exploration_rate