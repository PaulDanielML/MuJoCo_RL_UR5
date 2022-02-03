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
import copy
import math
from collections import deque, defaultdict
import time
from Modules import MULTIDISCRETE_RESNET


HEIGHT = 200
WIDTH = 200
N_EPISODES = 1000
STEPS_PER_EPISODE = 50
MEMORY_SIZE = 2000
MAX_POSSIBLE_SAMPLES = (
    12  # Number of transitions that fits on GPU memory for one backward-call (12 for RGB-D)
)
NUMBER_ACCUMULATIONS_BEFORE_UPDATE = 1  # How often to accumulate gradients before updating
BATCH_SIZE = MAX_POSSIBLE_SAMPLES * NUMBER_ACCUMULATIONS_BEFORE_UPDATE  # Effective batch size
GAMMA = 0.0
LEARNING_RATE = 0.001
EPS_STEADY = 0.0
EPS_START = 1.0
EPS_END = 0.2
EPS_DECAY = 8000
SAVE_WEIGHTS = True
MODEL = "RESNET"
ALGORITHM = "DQN"
OPTIMIZER = "ADAM"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Grasp_Agent:
    """
    Example class for an agent interacting with the 'GraspEnv'-environment.
    Implements some basic methods for normalization, action selection, observation transformation and learning.
    """

    def __init__(
        self,
        height=HEIGHT,
        width=WIDTH,
        learning_rate=LEARNING_RATE,
        mem_size=MEMORY_SIZE,
        eps_start=EPS_START,
        eps_end=EPS_END,
        eps_decay=EPS_DECAY,
        depth_only=False,
        load_path=None,
        train=True,
        seed=20,
        optimizer=OPTIMIZER,
    ):
        """
        Args:
            height: Observation height (in pixels).
            width: Observation width (in pixels).
            mem_size: Number of transitions to be stored in the replay buffer.
            eps_start, eps_end, eps_decay: Parameters describing the decay of epsilon.
            load_path: If training is to be resumed based on existing weights, they will be loaded from this path.
            train: If True, will be fully initialized, including replay buffer. Can be set to False for demonstration purposes.
        """

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.WIDTH = width
        self.HEIGHT = height
        self.depth_only = depth_only
        if train:
            self.env = gym.make(
                "gym_grasper:Grasper-v0", image_height=HEIGHT, image_width=WIDTH, render=False
            )
            # self.env = gym.make('gym_grasper:Grasper-v0', image_height=HEIGHT, image_width=WIDTH)
        else:
            self.env = gym.make(
                "gym_grasper:Grasper-v0",
                image_height=HEIGHT,
                image_width=WIDTH,
                show_obs=False,
                demo=True,
                render=True,
            )
        self.n_actions_1, self.n_actions_2 = (
            self.env.action_space.nvec[0],
            self.env.action_space.nvec[1],
        )
        self.output = self.n_actions_1 * self.n_actions_2
        # Initialize networks
        self.policy_net = MULTIDISCRETE_RESNET(number_actions_dim_2=self.n_actions_2).to(device)
        # Only need a target network if gamma is not zero
        if GAMMA != 0.0:
            self.target_net = MULTIDISCRETE_RESNET(number_actions_dim_2=self.n_actions_2).to(device)
            # No need for training on target net, we just copy the weigts from policy nets if we use it
            self.target_net.eval()
        # Load weights if training should not start from scratch
        if load_path is not None:
            checkpoint = torch.load(load_path)
            self.policy_net.load_state_dict(checkpoint["model_state_dict"])
            if GAMMA != 0.0:
                self.target_net.load_state_dict(checkpoint["model_state_dict"])
            print("Successfully loaded weights from {}.".format(load_path))
        # Set up some transforms
        self.normal_rgb = T.Compose(
            [
                T.ToPILImage(),
                T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                T.ToTensor(),
            ]
        )
        # self.normal_rgb = T.Compose([T.ToPILImage(), T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5), T.ToTensor(), \
        # T.Lambda(lambda x : x + 0.01*torch.randn_like(x))])
        self.normal_rgb_no_jitter_no_noise = T.Compose([T.ToTensor()])
        self.normal_depth = T.Compose([T.Lambda(lambda x: x + 0.01 * torch.randn_like(x))])
        # self.normal_depth =T.Compose([T.Lambda(lambda x : x + 0.001*torch.randn_like(x))])
        self.depth_threshold = np.round(
            self.env.model.cam_pos0[self.env.model.camera_name2id("top_down")][2]
            - self.env.TABLE_HEIGHT
            + 0.01,
            decimals=3,
        )
        self.last_action = None
        if train:
            # Set up replay buffer
            # TODO: Implement prioritized experience replay
            if GAMMA == 0.0:
                # Don't need to store the next state in the buffer if gamma is 0
                self.memory = ReplayBuffer(mem_size, simple=True)
            else:
                self.memory = ReplayBuffer(mem_size)
            if optimizer == "SGD":
                # Using SGD with parameters described in TossingBot paper
                self.optimizer = optim.SGD(
                    self.policy_net.parameters(),
                    lr=learning_rate,
                    momentum=0.9,
                    weight_decay=0.00002,
                )
            elif optimizer == "ADAM":
                self.optimizer = optim.Adam(
                    self.policy_net.parameters(), lr=learning_rate, weight_decay=0.00002
                )
            if load_path is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.steps_done = checkpoint["step"] if "step" in checkpoint.keys() else 0
                self.eps_threshold = (
                    checkpoint["epsilon"] if "epsilon" in checkpoint.keys() else EPS_STEADY
                )
                self.DESCRIPTION = "_continue_" + load_path[:-11] + "_at_" + str(self.steps_done)
                self.WEIGHT_PATH = load_path
                self.greedy_rotations = (
                    checkpoint["greedy_rotations"]
                    if "greedy_rotations" in checkpoint.keys()
                    else defaultdict(int)
                )
                self.greedy_rotations_successes = (
                    checkpoint["greedy_rotations_successes"]
                    if "greedy_rotations_successes" in checkpoint.keys()
                    else defaultdict(int)
                )
                self.random_rotations_successes = (
                    checkpoint["random_rotations_successes"]
                    if "random_rotations_successes" in checkpoint.keys()
                    else defaultdict(int)
                )
            else:
                self.steps_done = 0
                self.eps_threshold = EPS_START
                date = "_".join(
                    [
                        str(time.localtime()[1]),
                        str(time.localtime()[2]),
                        str(time.localtime()[0]),
                        str(time.localtime()[3]),
                        str(time.localtime()[4]),
                    ]
                )
                self.DESCRIPTION = "_".join(
                    [
                        ALGORITHM,
                        MODEL,
                        "LR",
                        str(learning_rate),
                        "OPTIM",
                        optimizer,
                        "H",
                        str(HEIGHT),
                        "W",
                        str(WIDTH),
                        "STEPS",
                        str(N_EPISODES * STEPS_PER_EPISODE),
                        "BUFFER_SIZE",
                        str(MEMORY_SIZE),
                        "BATCH_SIZE",
                        str(BATCH_SIZE),
                        "SEED",
                        str(seed),
                    ]
                )
                self.WEIGHT_PATH = self.DESCRIPTION + "_" + date + "_weights.pt"
                self.greedy_rotations = defaultdict(int)
                self.greedy_rotations_successes = defaultdict(int)
                self.random_rotations_successes = defaultdict(int)
            # Tensorboard setup
            self.writer = SummaryWriter(comment=self.DESCRIPTION)
            if not self.depth_only:
                self.writer.add_graph(
                    self.policy_net, torch.zeros(1, 4, self.WIDTH, self.HEIGHT).to(device)
                )
            else:
                self.writer.add_graph(
                    self.policy_net, torch.zeros(1, 1, self.WIDTH, self.HEIGHT).to(device)
                )
            self.last_1000_rewards = deque(maxlen=1000)
            self.last_100_loss = deque(maxlen=100)
            self.last_1000_actions = deque(maxlen=1000)

    def epsilon_greedy(self, state):
        """
        Returns an action according to the epsilon-greedy policy.

        Args:
            state: An observation / state that will be forwarded through the policy net if greedy action is chosen.
        """

        sample = random.random()
        self.eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
            -1.0 * self.steps_done / EPS_DECAY
        )
        # self.eps_threshold = EPS_STEADY
        self.writer.add_scalar("Epsilon", self.eps_threshold, global_step=self.steps_done)
        self.steps_done += 1
        # if self.steps_done < 2*BATCH_SIZE:
        # self.last_action = 'random'
        # return torch.tensor([[random.randrange(self.output)]], dtype=torch.long)
        if sample > self.eps_threshold:
            self.last_action = "greedy"
            with torch.no_grad():
                # For RESNET
                max_idx = self.policy_net(state.to(device)).view(-1).max(0)[1]
                max_idx = max_idx.view(1)
                # Do not want to store replay buffer in GPU memory, so put action tensor to cpu.
                return max_idx.unsqueeze_(0).cpu()
        # else:
        #     self.last_action = 'random'
        #     return torch.tensor([[random.randrange(self.output)]], dtype=torch.long)

        # Little trick for faster training: When sampling a random action, check the depth value
        # of the selected pixel and resample until you get a pixel corresponding to a point on the table
        else:
            self.last_action = "random"
            while True:
                action = random.randrange(self.output)
                action_1 = action % self.n_actions_1
                x = action_1 % self.env.IMAGE_WIDTH
                y = action_1 // self.env.IMAGE_WIDTH
                depth = self.env.current_observation["depth"][y][x]
                coordinates = self.env.controller.pixel_2_world(
                    pixel_x=x,
                    pixel_y=y,
                    depth=depth,
                    height=self.env.IMAGE_HEIGHT,
                    width=self.env.IMAGE_WIDTH,
                )
                if coordinates[2] >= (self.env.TABLE_HEIGHT - 0.01):
                    break

            return torch.tensor([[action]], dtype=torch.long)

    def greedy(self, state):
        """
        Always returns the greedy action. For demonstrating learned behaviour.

        Args:
            state: An observation / state that will be forwarded through the policy network to receive the action with the highest Q value.
        """

        self.last_action = "greedy"

        with torch.no_grad():
            max_o = self.policy_net(state.to(device)).view(-1).max(0)
            max_idx = max_o[1]
            max_value = max_o[0]

            return max_idx, max_value.item()

    def transform_observation(self, observation, normalize=True, jitter_and_noise=True):
        """
        Takes an observation dictionary, transforms it into a normalized tensor of shape (1,4,height,width).
        The returned tensor will already be on the gpu if one is available.
        NEW: Also adds some random noise to the input.

        Args:
            observation: Observation to be transformed.
        """

        depth = copy.deepcopy(observation["depth"])
        depth[np.where(depth > self.depth_threshold)] = self.depth_threshold

        if normalize:
            if not self.depth_only:
                rgb = copy.deepcopy(observation["rgb"])

            depth += np.random.normal(loc=0, scale=0.001, size=depth.shape)
            depth *= -1
            depth_min = np.min(depth)
            depth_max = np.max(depth)
            depth = (depth - depth_min) / (depth_max - depth_min)
        else:
            rgb = observation["rgb"].astype(np.float32)

        # Add channel dimension to np-array depth.
        depth = np.expand_dims(depth, 0)
        # Apply rgb normalization transform, this rearanges dimensions, transforms into float tensor,
        # scales values to range [0,1]
        if not self.depth_only:
            if normalize and jitter_and_noise:
                rgb_tensor = self.normal_rgb(rgb).float()
            if normalize and not jitter_and_noise:
                rgb_tensor = self.normal_rgb_no_jitter_no_noise(rgb).float()
            if not normalize:
                # Read in the means and stds from another file, created by 'normalize.py'
                self.means, self.stds = self.get_mean_std()
                self.standardize_rgb = T.Compose(
                    [T.ToTensor(), T.Normalize(self.means[0:3], self.stds[0:3])]
                )
                rgb_tensor = self.standardize_rgb(rgb).float()

        depth_tensor = torch.tensor(depth).float()
        # Depth values need to be normalized separately, as they are not int values. Therefore, T.ToTensor() does not work for them.
        # if normalize:
        # depth_tensor = self.normal_depth(depth_tensor)
        if not normalize:
            self.standardize_depth = T.Compose(
                [
                    T.Normalize(self.means[3], self.stds[3]),
                    T.Lambda(lambda x: x + 0.001 * torch.randn_like(x)),
                ]
            )
            depth_tensor = self.standardize_depth(depth_tensor)

        if not self.depth_only:
            obs_tensor = torch.cat((rgb_tensor, depth_tensor), dim=0)
        else:
            obs_tensor = depth_tensor.detach().clone()

        # Add batch dimension.
        obs_tensor.unsqueeze_(0)
        if not self.depth_only:
            del rgb, depth, rgb_tensor, depth_tensor
        else:
            del depth, depth_tensor

        return obs_tensor

    def get_mean_std(self):
        """
        Reads and returns the mean and standard deviation values created by 'normalize.py'.
        """

        with open("mean_and_std", "rb") as file:
            raw = file.read()
            values = pickle.loads(raw)

        return values[0:4], values[4:8]

    def transform_action(self, action):
        action_value = action.item()
        action_1 = action_value % self.n_actions_1
        action_2 = action_value // self.n_actions_1

        return np.array([action_1, action_2])

    def learn(self):
        """
        Example implementaion of a training method, using standard DQN-learning.
        Samples batches from the replay buffer, feeds them through the policy net, calculates loss,
        and calls the optimizer.
        """

        # Make sure we have collected enough data for at least one batch
        if len(self.memory) < 2 * BATCH_SIZE:
            print("Filling the replay buffer ...")
            return

        # Sample the replay buffer
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch for easier access (see https://stackoverflow.com/a/19343/3343043)
        if GAMMA == 0.0:
            batch = simple_Transition(*zip(*transitions))
        else:
            batch = Transition(*zip(*transitions))

        # Gradient accumulation to bypass GPU memory restrictions
        for i in range(NUMBER_ACCUMULATIONS_BEFORE_UPDATE):
            # Transfer weights every TARGET_NETWORK_UPDATE steps
            if GAMMA != 0.0:
                if self.steps_done % TARGET_NETWORK_UPDATE == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

            start_idx = i * MAX_POSSIBLE_SAMPLES
            end_idx = (i + 1) * MAX_POSSIBLE_SAMPLES

            state_batch = torch.cat(batch.state[start_idx:end_idx]).to(device)
            action_batch = torch.cat(batch.action[start_idx:end_idx]).to(device)
            if GAMMA != 0.0:
                next_state_batch = torch.cat(batch.next_state[start_idx:end_idx]).to(device)
            reward_batch = torch.cat(batch.reward[start_idx:end_idx]).to(device)

            # Current Q prediction of our policy net, for the actions we took
            q_pred = (
                self.policy_net(state_batch).view(MAX_POSSIBLE_SAMPLES, -1).gather(1, action_batch)
            )
            # q_pred = self.policy_net(state_batch).gather(1, action_batch)

            if GAMMA == 0.0:
                q_expected = reward_batch.float()
            else:
                # Q prediction of the target net of the next state
                q_next_state = self.target_net(next_state_batch).max(1)[0].unsqueeze(1).detach()

                # Calulate expected Q value using Bellmann: Q_t = r + gamma*Q_t+1
                q_expected = reward_batch + (GAMMA * q_next_state)

            loss = F.binary_cross_entropy(q_pred, q_expected) / NUMBER_ACCUMULATIONS_BEFORE_UPDATE
            loss.backward()

        self.last_100_loss.append(loss.item())
        # self.writer.add_scalar('Average loss', loss, global_step=self.steps_done)
        self.optimizer.step()

        self.optimizer.zero_grad()

    def update_tensorboard(self, reward, action):
        """
        Method for keeping track of tensorboard metrics.

        Args:
            reward: Reward to be added to the list of last 1000 rewards.
            action: Last action chosen by the current policy.
        """

        rotation_action = action[1]
        self.last_1000_actions.append(rotation_action)
        if self.last_action == "greedy":
            self.greedy_rotations[str(rotation_action)] += 1
            if reward == 1:
                self.greedy_rotations_successes[str(rotation_action)] += 1
        else:
            if reward == 1:
                self.random_rotations_successes[str(rotation_action)] += 1

        if self.steps_done % 1000 == 0:
            self.writer.add_histogram(
                "Rotation action distribution/Last1000",
                np.array(self.last_1000_actions),
                global_step=self.steps_done,
                bins=[i for i in range(self.n_actions_2)],
            )

        if self.steps_done % 10 == 0:
            self.writer.add_scalars(
                "Total number of rotation actions/Greedy", self.greedy_rotations, self.steps_done
            )
            self.writer.add_scalars(
                "Total number of successful rotation actions/Greedy",
                self.greedy_rotations_successes,
                self.steps_done,
            )
            self.writer.add_scalars(
                "Total number of successful rotation actions/Random",
                self.random_rotations_successes,
                self.steps_done,
            )

        self.last_1000_rewards.append(reward)

        if len(self.last_1000_rewards) > 99:
            if self.steps_done % 10 == 0:
                last_100 = np.array([self.last_1000_rewards[i] for i in range(-100, 0)])
                mean_reward_100 = np.mean(last_100)
                self.writer.add_scalar(
                    "Mean reward/Last100", mean_reward_100, global_step=self.steps_done
                )
            # grasps_in_last_100 = np.count_nonzero(last_100 == 1)
            # self.writer.add_scalar('Number of succ. grasps in last 100 steps', grasps_in_last_100, global_step=self.steps_done)
        if len(self.last_1000_rewards) > 999:
            if self.steps_done % 10 == 0:
                mean_reward_1000 = np.mean(self.last_1000_rewards)
                self.writer.add_scalar(
                    "Mean reward/Last1000", mean_reward_1000, global_step=self.steps_done
                )

        if len(self.last_100_loss) > 99:
            if self.steps_done % 10 == 0:
                self.writer.add_scalar(
                    "Mean loss/Last100", np.mean(self.last_100_loss), global_step=self.steps_done
                )


def main():

    for rand_seed in [999]:
        for lr in [0.0005]:
            LOAD_PATH = "DQN_RESNET_LR_0.001_OPTIM_ADAM_H_200_W_200_STEPS_35000_BUFFER_SIZE_2000_BATCH_SIZE_12_SEED_81_9_7_2020_9_52_weights.pt"

            agent = Grasp_Agent(seed=rand_seed, load_path=None, learning_rate=lr, depth_only=False)
            agent.optimizer.zero_grad()
            for episode in range(1, N_EPISODES + 1):
                state = agent.env.reset()
                state = agent.transform_observation(state)
                print(
                    colored(
                        "CURRENT EPSILON: {}".format(agent.eps_threshold),
                        color="blue",
                        attrs=["bold"],
                    )
                )
                for step in range(1, STEPS_PER_EPISODE + 1):
                    print("#################################################################")
                    print(
                        colored(
                            "EPISODE {} STEP {}".format(episode, step),
                            color="white",
                            attrs=["bold"],
                        )
                    )
                    print("#################################################################")
                    action = agent.epsilon_greedy(state)
                    env_action = agent.transform_action(action)
                    next_state, reward, done, _ = agent.env.step(
                        env_action, action_info=agent.last_action
                    )
                    agent.update_tensorboard(reward, env_action)
                    reward = torch.tensor([[reward]])
                    next_state = agent.transform_observation(next_state)
                    if GAMMA == 0.0:
                        agent.memory.push(state, action, reward)
                    else:
                        agent.memory.push(state, action, next_state, reward)

                    state = next_state

                    agent.learn()

            if SAVE_WEIGHTS:
                torch.save(
                    {
                        "step": agent.steps_done,
                        "model_state_dict": agent.policy_net.state_dict(),
                        "optimizer_state_dict": agent.optimizer.state_dict(),
                        "epsilon": agent.eps_threshold,
                        "greedy_rotations": agent.greedy_rotations,
                        "greedy_rotations_successes": agent.greedy_rotations_successes,
                        "random_rotations_successes": agent.random_rotations_successes,
                    },
                    agent.WEIGHT_PATH,
                )

                # torch.save(agent.policy_net.state_dict(), WEIGHT_PATH)
                print("Saved checkpoint to {}.".format(agent.WEIGHT_PATH))

            print(f"Finished training (rand_seed = {rand_seed}).")
            agent.writer.close()
            agent.env.close()


if __name__ == "__main__":
    main()
