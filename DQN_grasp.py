import matplotlib
import matplotlib.pyplot as plt
import gym
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import torch.optim as optim
from DQN import DQN, ReplayBuffer, Transition
from termcolor import colored
import numpy as np
import pickle
import random
import math

HEIGHT = 200
WIDTH = 200
N_EPISODES = 50
STEPS_PER_EPISODE = 50
TARGET_NETWORK_UPDATE = 100
MEMORY_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
SAVE_WEIGHTS = True
LOAD_WEIGHTS = False
WEIGHT_PATH = './{}_{}_weights'.format(HEIGHT, WIDTH)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plt.ion()

class DQN_Agent():
    def __init__(self):
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.output = self.WIDTH * self.HEIGHT
        self.policy_net = DQN(self.WIDTH, self.HEIGHT, self.output).to(device)
        self.target_net = DQN(self.WIDTH, self.HEIGHT, self.output).to(device)
        if LOAD_WEIGHTS:
            self.policy_net.load_state_dict(torch.load(WEIGHT_PATH))
            self.target_net.load_state_dict(torch.load(WEIGHT_PATH))
        self.target_net.eval()
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.learn_step_counter = 0
        self.means, self.stds = self.get_mean_std()
        self.normal = T.Compose([T.ToTensor(), T.Normalize(self.means, self.stds)])
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.steps_done = 0

    def epsilon_greedy(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
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


    def get_max_index(self, tensor):
        assert isinstance(tensor, torch.Tensor)
        with torch.no_grad():
            return tensor.max(1)[1][0]


    def get_mean_std(self):
        with open('mean_and_std', 'rb') as file:
            raw = file.read()
            values = pickle.loads(raw)

        return values[0:3], values[3:6]





    def learn(self):

        if len(self.memory) < BATCH_SIZE:
            print('Filling the replay buffer ...')
            return

        if self.steps_done % TARGET_NETWORK_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        next_state_batch = torch.cat(batch.next_state)
        reward_batch = torch.cat(batch.reward)

        # Current Q prediction of our policy net, for the actions we took
        q_pred = self.policy_net(state_batch).gather(1, action_batch)

        # Q prediction of the target net of the next state
        # print(reward_batch.size())
        # print(self.target_net(next_state_batch).max(1)[0].unsqueeze(1).size())
        # print((self.target_net(next_state_batch).max(1)[0].unsqueeze(1) * GAMMA) + reward_batch)
        q_next_state = self.target_net(next_state_batch).max(1)[0].unsqueeze(1).detach()

        # Calulate expected Q value using Bellmann: Q_t = r + gamma*Q_t+1
        q_expected = reward_batch + (GAMMA * q_next_state)

        loss = F.smooth_l1_loss(q_pred, q_expected)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


rewards_list = []

def plot_rewards():
    plt.figure(2)
    plt.clf()
    rewards_t = torch.tensor(rewards_list, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Time-step')
    plt.ylabel('Reward')
    plt.plot(rewards_t.numpy())
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

def main():
    env = gym.make('gym_grasper:Grasper-v0', image_height=HEIGHT, image_width=WIDTH)
    agent = DQN_Agent()
    for episode in range(1, N_EPISODES+1):
        state = env.reset()
        state = agent.transform_observation(state)
        for step in range(1, STEPS_PER_EPISODE+1):
            print('#################################################################')
            print(colored('EPISODE {} STEP {}'.format(episode, step), color='white', attrs=['bold']))
            print('#################################################################')
            action = agent.epsilon_greedy(state)
            # action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action.item())
            rewards_list.append(reward)
            # next_state, reward, done, _ = env.step(action)
            action = torch.tensor([[action]], device=device, dtype=torch.long) 
            reward = torch.tensor([[reward]], device=device)
            next_state = agent.transform_observation(next_state)
            agent.memory.push(state, action, next_state, reward)


            state = next_state

            agent.learn()

            plot_rewards()

    if SAVE_WEIGHTS:
        torch.save(agent.policy_net.state_dict(), WEIGHT_PATH)
        print('Saved weights to {}.'.format(WEIGHT_PATH))


    print('Finished training.')
    env.close()
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()


