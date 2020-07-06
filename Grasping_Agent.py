# Author: Paul Daniel (pdd@mp.aau.dk)

import gym
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from Modules import ReplayBuffer, Transition
from termcolor import colored
import numpy as np
import pickle
import random
import math
from collections import deque

HEIGHT = 200
WIDTH = 200
N_EPISODES = 200
STEPS_PER_EPISODE = 50
TARGET_NETWORK_UPDATE = 50
MEMORY_SIZE = 1000
BATCH_SIZE = 10
GAMMA = 0.0
LEARNING_RATE = 0.001
EPS_START = 0.5
EPS_END = 0.1
EPS_DECAY = 4000
SAVE_WEIGHTS = True
LOAD_WEIGHTS = False
BUFFER = 'RBSTANDARD'
# MODEL = 'CONV3_FC1'
MODEL = 'RESNET'

DESCRIPTION = '_'.join([MODEL, BUFFER, 'LR', str(LEARNING_RATE), 'H', str(HEIGHT), \
                'W', str(WIDTH), 'STEPS', str(N_EPISODES*STEPS_PER_EPISODE)])

WEIGHT_PATH = DESCRIPTION + '_weights.pt'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Grasp_Agent():
    def __init__(self, height=HEIGHT, width=WIDTH, mem_size=MEMORY_SIZE, eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY, load_path=None):
        self.WIDTH = width
        self.HEIGHT = height
        self.output = self.WIDTH * self.HEIGHT
        if MODEL == 'CONV3_FC1':
            from Modules import CONV3_FC1
            self.policy_net = CONV3_FC1(self.WIDTH, self.HEIGHT, self.output).to(device)
        elif MODEL == 'RESNET':
            from Modules import RESNET
            self.policy_net = RESNET().to(device)
        if GAMMA != 0.0:
            if MODEL == 'CONV3_FC1':
                self.target_net = CONV3_FC1(self.WIDTH, self.HEIGHT, self.output).to(device)
            elif MODEL == 'RESNET':
                self.target_net = RESNET().to(device)
            self.target_net.eval()
        if load_path is not None:
            self.policy_net.load_state_dict(torch.load(load_path))
            if GAMMA != 0.0:
                self.target_net.load_state_dict(torch.load(WEIGHT_PATH))
            print('Successfully loaded weights from {}.'.format(WEIGHT_PATH))
        self.memory = ReplayBuffer(mem_size)
        self.means, self.stds = self.get_mean_std()
        self.normal = T.Compose([T.ToTensor(), T.Normalize(self.means, self.stds)])
        # self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.optimizer = optim.SGD(self.policy_net.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.00002)
        self.steps_done = 0
        self.eps_threshold = EPS_START
        self.writer = SummaryWriter(comment=DESCRIPTION)
        self.writer.add_graph(self.policy_net, torch.zeros(1, 3, self.WIDTH, self.HEIGHT).to(device))
        self.writer.close()
        self.last_1000_rewards = deque(maxlen=1000)


    def epsilon_greedy(self, state):
        sample = random.random()
        self.eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        self.writer.add_scalar('Epsilon', self.eps_threshold, global_step=self.steps_done)
        if sample > self.eps_threshold:
            with torch.no_grad():
                # For RESNET
                max_idx = self.policy_net(state).view(-1).max(0)[1]
                max_idx = max_idx.view(1)
                return max_idx.unsqueeze_(0)
                # return self.policy_net(state).max(1)[1].view(1, 1) # For CONV3_FC1
        else:
            return torch.tensor([[random.randrange(self.output)]], device=device, dtype=torch.long)


    def greedy(self, state):
        with torch.no_grad():
            # For RESNET
            max_idx = self.policy_net(state).view(-1).max(0)[1]
            max_idx = max_idx.view(1)
            return max_idx.unsqueeze_(0)


    def transform_observation(self, observation):

        # For now: only use the rgb data of the observation
        obs = observation['rgb']

        # Apply transform, this rearanges dimensions, transforms into float tensor,
        # scales values to range [0,1] and normalizes data, sends to gpu if available
        obs_tensor = self.normal(obs).float().to(device)
        # Add batch dimension
        obs_tensor.unsqueeze_(0)

        return obs_tensor


    def get_mean_std(self):
        """
        Reads and returns the mean and standard deviation values creates by 'normalize.py'. Currently only rgb values are returned.
        """

        with open('mean_and_std', 'rb') as file:
            raw = file.read()
            values = pickle.loads(raw)

        return values[0:3], values[3:6]


    def learn(self):

        # Make sure we have collected enough data for at least one batch
        if len(self.memory) < BATCH_SIZE:
            print('Filling the replay buffer ...')
            return

        # Transfer weights every TARGET_NETWORK_UPDATE steps
        if GAMMA != 0.0:
            if self.steps_done % TARGET_NETWORK_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        # Sample the replay buffer
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch for easier access (see https://stackoverflow.com/a/19343/3343043)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        next_state_batch = torch.cat(batch.next_state)
        reward_batch = torch.cat(batch.reward)

        # Current Q prediction of our policy net, for the actions we took
        q_pred = self.policy_net(state_batch).view(BATCH_SIZE, -1).gather(1, action_batch)
        # q_pred = self.policy_net(state_batch).gather(1, action_batch)

        if GAMMA == 0.0:
            q_expected = reward_batch.float()

        else:
            # Q prediction of the target net of the next state
            q_next_state = self.target_net(next_state_batch).max(1)[0].unsqueeze(1).detach()

            # Calulate expected Q value using Bellmann: Q_t = r + gamma*Q_t+1
            q_expected = reward_batch + (GAMMA * q_next_state)

        # loss = F.smooth_l1_loss(q_pred, q_expected)
        loss = F.binary_cross_entropy(q_pred, q_expected)

        # loss = F.binary_cross_entropy(self.output_unit(q_pred), q_expected)

        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.policy_net.parameters():
            # param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_tensorboard(self, reward):
        self.last_1000_rewards.append(reward)

        if len(self.last_1000_rewards) > 100: 
            last_100 = np.array([self.last_1000_rewards[i] for i in range(-100,0)])
            mean_reward_100 = np.mean(last_100)
            self.writer.add_scalar('Mean reward/Last100', mean_reward_100, global_step=self.steps_done)
            grasps_in_last_100 = np.count_nonzero(last_100 == 1)
            self.writer.add_scalar('Number of succ. grasps in last 100 steps', grasps_in_last_100, global_step=self.steps_done)
        if len(self.last_1000_rewards) > 999:
            mean_reward_1000 = np.mean(self.last_1000_rewards)
            self.writer.add_scalar('Mean reward/Last1000', mean_reward_1000, global_step=self.steps_done)


def main():
    env = gym.make('gym_grasper:Grasper-v0', image_height=HEIGHT, image_width=WIDTH)
    agent = Grasp_Agent()
    for episode in range(1, N_EPISODES+1):
        state = env.reset()
        state = agent.transform_observation(state)
        print(colored('CURRENT EPSILON: {}'.format(agent.eps_threshold), color='blue', attrs=['bold']))
        for step in range(1, STEPS_PER_EPISODE+1):
            print('#################################################################')
            print(colored('EPISODE {} STEP {}'.format(episode, step), color='white', attrs=['bold']))
            print('#################################################################')
            
            action = agent.epsilon_greedy(state)
            next_state, reward, done, _ = env.step(action.item())
            agent.update_tensorboard(reward)
            reward = torch.tensor([[reward]], device=device)
            next_state = agent.transform_observation(next_state)
            agent.memory.push(state, action, next_state, reward)


            state = next_state

            agent.learn()

    if SAVE_WEIGHTS:
        torch.save(agent.policy_net.state_dict(), WEIGHT_PATH)
        print('Saved weights to {}.'.format(WEIGHT_PATH))


    print('Finished training.')
    agent.writer.close()
    env.close()

if __name__ == '__main__':
    main()


