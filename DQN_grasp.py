import gym
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from DQN import DQN, ReplayBuffer, Transition
from termcolor import colored
import numpy as np
import pickle
import random
import math
from collections import deque

HEIGHT = 200
WIDTH = 200
N_EPISODES = 3000
STEPS_PER_EPISODE = 50
TARGET_NETWORK_UPDATE = 50
MEMORY_SIZE = 1000
BATCH_SIZE = 64
GAMMA = 0.0
LEARNING_RATE = 0.01
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 1500
SAVE_WEIGHTS = True
LOAD_WEIGHTS = False
WEIGHT_PATH = './{}_{}_weights.pt'.format(HEIGHT, WIDTH)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DQN_Agent():
    def __init__(self):
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.output = self.WIDTH * self.HEIGHT
        self.policy_net = DQN(self.WIDTH, self.HEIGHT, self.output).to(device)
        if GAMMA != 0.0:
            self.target_net = DQN(self.WIDTH, self.HEIGHT, self.output).to(device)
            self.target_net.eval()
        if LOAD_WEIGHTS:
            self.policy_net.load_state_dict(torch.load(WEIGHT_PATH))
            if GAMMA != 0.0:
                self.target_net.load_state_dict(torch.load(WEIGHT_PATH))
            print('Successfully loaded weights from {}.'.format(WEIGHT_PATH))
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.learn_step_counter = 0
        self.means, self.stds = self.get_mean_std()
        self.normal = T.Compose([T.ToTensor(), T.Normalize(self.means, self.stds)])
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        # self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.steps_done = 0
        self.eps_threshold = EPS_START
        self.writer = SummaryWriter()
        self.last_100_rewards = deque(maxlen=100)
        self.grasps = 0


    def epsilon_greedy(self, state):
        sample = random.random()
        self.eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        self.writer.add_scalar('Epsilon', self.eps_threshold, global_step=self.steps_done)
        self.writer.close()
        if sample > self.eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.output)]], device=device, dtype=torch.long)


    def transform_observation(self, observation):
        # Transform observation to tensor
        # obs_tensor = torch.from_numpy(observation['rgb']).float().to(device)
        obs_tensor = observation['rgb']
        # Rearrange to match Torch input format
        # obs_tensor = obs_tensor.permute(2,0,1)

        obs_tensor = self.normal(obs_tensor).float().to(device)
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
        q_pred = self.policy_net(state_batch).gather(1, action_batch)

        if GAMMA == 0.0:
            q_expected = reward_batch.float()

        else:
            # Q prediction of the target net of the next state
            q_next_state = self.target_net(next_state_batch).max(1)[0].unsqueeze(1).detach()

            # Calulate expected Q value using Bellmann: Q_t = r + gamma*Q_t+1
            q_expected = reward_batch + (GAMMA * q_next_state)

        loss = F.smooth_l1_loss(q_pred, q_expected)

        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.policy_net.parameters():
            # param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_tensorboard(self, reward):
        self.last_100_rewards.append(reward)
        mean_reward = np.mean(self.last_100_rewards)
        self.writer.add_scalar('Mean reward/Last100', mean_reward, global_step=self.steps_done)
        if reward == 100:
            self.grasps += 1
            self.writer.add_scalar('Total Successful Grasps', self.grasps, global_step=self.steps_done)
        self.writer.close()


def main():
    env = gym.make('gym_grasper:Grasper-v0', image_height=HEIGHT, image_width=WIDTH)
    agent = DQN_Agent()
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
    env.close()

if __name__ == '__main__':
    main()


