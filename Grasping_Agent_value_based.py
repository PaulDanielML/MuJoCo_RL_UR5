# Author: Paul Daniel (pdd@mp.aau.dk)

import gym
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from Modules import ReplayBuffer, Transition, simple_Transition
from termcolor import colored
import numpy as np
import pickle
import random
import math
from collections import deque
import time

HEIGHT = 200
WIDTH = 200
N_EPISODES = 2000
STEPS_PER_EPISODE = 5
TARGET_NETWORK_UPDATE = 50
MEMORY_SIZE = 900
BATCH_SIZE = 10
GAMMA = 0.0
LEARNING_RATE = 0.001
EPS_START = 0.5
EPS_END = 0.1
EPS_DECAY = 4000
SAVE_WEIGHTS = True
BUFFER = 'RBSTANDARD'
MODEL = 'RESNET'

date = '_'.join([str(time.localtime()[1]), str(time.localtime()[2]), str(time.localtime()[0]), str(time.localtime()[3]), str(time.localtime()[4])])


DESCRIPTION = '_'.join([MODEL, BUFFER, 'LR', str(LEARNING_RATE), 'H', str(HEIGHT), \
                'W', str(WIDTH), 'STEPS', str(N_EPISODES*STEPS_PER_EPISODE)])

WEIGHT_PATH = DESCRIPTION + '_' + date + '_weights.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Grasp_Agent():
    """
    Example class for an agent interacting with the 'GraspEnv'-environment. 
    Implements some basic methods for normalization, action selection, observation transformation and learning. 
    """

    def __init__(self, height=HEIGHT, width=WIDTH, mem_size=MEMORY_SIZE, eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY, load_path=None, train=True):
        """
        Args:
            height: Observation height (in pixels).
            width: Observation width (in pixels).
            mem_size: Number of transitions to be stored in the replay buffer.
            eps_start, eps_end, eps_decay: Parameters describing the decay of epsilon.
            load_path: If training is to be resumed based on existing weights, they will be loaded from this path.
            train: If True, will be fully initialized, including replay buffer. Can be set to False for demonstration purposes.
        """

        self.WIDTH = width
        self.HEIGHT = height
        self.output = self.WIDTH * self.HEIGHT
        # Initialize networks
        if MODEL == 'CONV3_FC1':
            from Modules import CONV3_FC1
            self.policy_net = CONV3_FC1(self.WIDTH, self.HEIGHT, self.output).to(device)
        elif MODEL == 'RESNET':
            from Modules import RESNET
            self.policy_net = RESNET().to(device)
        # Only need a target network if gamma is not zero
        if GAMMA != 0.0:
            if MODEL == 'CONV3_FC1':
                self.target_net = CONV3_FC1(self.WIDTH, self.HEIGHT, self.output).to(device)
            elif MODEL == 'RESNET':
                self.target_net = RESNET().to(device)
            # No need for training on target net, we just copy the weigts from policy nets if we use it
            self.target_net.eval()
        # Load weights if training should not start from scratch
        if load_path is not None:
            self.policy_net.load_state_dict(torch.load(load_path))
            if GAMMA != 0.0:
                self.target_net.load_state_dict(torch.load(WEIGHT_PATH))
            print('Successfully loaded weights from {}.'.format(WEIGHT_PATH))
        # Read in the means and stds from another file, created by 'normalize.py'
        self.means, self.stds = self.get_mean_std()
        # Set up some transforms
        self.normal_rgb = T.Compose([T.ToTensor(), T.Normalize(self.means[0:3], self.stds[0:3])])
        self.normal_depth = T.Normalize(self.means[3], self.stds[3])
        if train:
            # Set up replay buffer
            # TODO: Implement prioritized experience replay
            if GAMMA == 0.0:
                # Don't need to store the next state in the buffer if gamma is 0
                self.memory = ReplayBuffer(mem_size, simple=True)
            else:
                self.memory = ReplayBuffer(mem_size)
            # Using SGD with parameters described in TossingBot paper
            self.optimizer = optim.SGD(self.policy_net.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.00002)
            self.steps_done = 0
            self.eps_threshold = EPS_START
            # Tensorboard setup
            self.writer = SummaryWriter(comment=DESCRIPTION)
            self.writer.add_graph(self.policy_net, torch.zeros(1, 4, self.WIDTH, self.HEIGHT).to(device))
            self.writer.close()
            self.last_1000_rewards = deque(maxlen=1000)


    def epsilon_greedy(self, state):
        """
        Returns an action according to the epsilon-greedy policy.

        Args:
            state: An observation / state that will be forwarded through the policy net if greedy action is chosen.
        """

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
        """
        Always returns the greedy action. For demonstrating learned behaviour. 

        Args: 
            state: An observation / state that will be forwarded through the policy to receive the action with the highest Q value. 
        """

        with torch.no_grad():
            max_o = self.policy_net(state).view(-1).max(0)
            max_idx = max_o[1]
            max_value = max_o[0]

            return max_idx.item(), max_value.item()


    def transform_observation(self, observation):
        """
        Takes an observation dictionary, transforms it into a normalized tensor of shape (1,4,height,width).
        The returned tensor will already be on the gpu if one is available. 

        Args:
            observation: Observation to be transformed.
        """

        rgb = observation['rgb']
        depth = observation['depth']
        # Add channel dimension to np-array depth.
        depth = np.expand_dims(depth, 0)
        # Apply rgb normalization transform, this rearanges dimensions, transforms into float tensor,
        # scales values to range [0,1] and normalizes data, sends to gpu if available.
        rgb_tensor = self.normal_rgb(rgb).float()
        depth_tensor = torch.tensor(depth).float()
        # Depth values need to be normalized separately, as they are not int values. Therefore, T.ToTensor() does not work for them.
        depth_tensor = self.normal_depth(depth_tensor)
        
        obs_tensor = torch.cat((rgb_tensor, depth_tensor), dim=0).to(device)

        # Add batch dimension.
        obs_tensor.unsqueeze_(0)
        del rgb, depth, rgb_tensor, depth_tensor

        return obs_tensor


    def transform_observation_rgb_only(self, observation):
        """
        Takes an observation dictionary, transforms it into a normalized tensor of shape (1,3,height,width), containing 
        only the rgb-values of the observation. 

        Args:
            observation: Observation to be transformed.
        """

        obs = observation['rgb']

        # Apply transform, this rearanges dimensions, transforms into float tensor,
        # scales values to range [0,1] and normalizes data, sends to gpu if available
        obs_tensor = self.normal_rgb(obs).float().to(device)
        # Add batch dimension
        obs_tensor.unsqueeze_(0)

        return obs_tensor


    def get_mean_std(self):
        """
        Reads and returns the mean and standard deviation values created by 'normalize.py'.
        """

        with open('mean_and_std', 'rb') as file:
            raw = file.read()
            values = pickle.loads(raw)

        return values[0:4], values[4:8]


    def learn(self):
        """
        Example implementaion of a training method, using standard DQN-learning.
        Samples batches from the replay buffer, feeds them through the policy net, calculates loss,
        and calls the optimizer. 
        """

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
        if GAMMA == 0.0:
            batch = simple_Transition(*zip(*transitions))
        else:
            batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        if GAMMA != 0.0:
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

        loss = F.binary_cross_entropy(q_pred, q_expected)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_tensorboard(self, reward):
        """
        Method for keeping track of the running reward averages.

        Args:  
            reward: Reward to be added to the list of last 1000 rewards.
        """
        
        self.last_1000_rewards.append(reward)

        if len(self.last_1000_rewards) > 100: 
            last_100 = np.array([self.last_1000_rewards[i] for i in range(-100,0)])
            mean_reward_100 = np.mean(last_100)
            self.writer.add_scalar('Mean reward/Last100', mean_reward_100, global_step=self.steps_done)
            # grasps_in_last_100 = np.count_nonzero(last_100 == 1)
            # self.writer.add_scalar('Number of succ. grasps in last 100 steps', grasps_in_last_100, global_step=self.steps_done)
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
            if GAMMA == 0.0:
                agent.memory.push(state, action, reward)
            else:
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
