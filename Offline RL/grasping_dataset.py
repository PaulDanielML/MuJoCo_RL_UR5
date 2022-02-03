import torch
from torch.utils.data import Dataset
import sys

sys.path.append("../")
from Modules import simple_Transition
import torchvision.transforms as T
import copy
import numpy as np


class Grasping_Dataset(Dataset):
    def __init__(self, file):
        data = torch.load(file)
        self.state_list = data["states"]
        self.action_list = data["actions"]
        self.reward_list = data["rewards"]
        self.normal_rgb = T.Compose(
            [
                T.ToPILImage(),
                T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                T.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.state_list)

    def __getitem__(self, idx):
        return [
            self.transform_observation(self.state_list[idx]),
            self.action_list[idx],
            self.reward_list[idx],
        ]
        # return simple_Transition(self.state_list[idx], self.action_list[idx], self.reward_list[idx])
        # return_list = []
        # for i in idx:
        # return_list.append(simple_Transition(self.state_list[i], self.action_list[i], self.reward_list[i]))

        # return return_list

    def transform_observation(self, observation, normalize=True, jitter_and_noise=True):

        depth = copy.deepcopy(observation["depth"])
        depth_threshold = 1.1
        depth[np.where(depth > depth_threshold)] = depth_threshold

        if normalize:
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
        if normalize and jitter_and_noise:
            rgb_tensor = self.normal_rgb(rgb).float()

        depth_tensor = torch.tensor(depth).float()

        obs_tensor = torch.cat((rgb_tensor, depth_tensor), dim=0)

        # Add batch dimension.
        # obs_tensor.unsqueeze_(0)
        del rgb, depth, rgb_tensor, depth_tensor

        return obs_tensor
