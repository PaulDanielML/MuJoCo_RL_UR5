import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from prettytable import PrettyTable
from collections import namedtuple
import numpy as np
import random
import time

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

def Transform_Image(means, stds):
	return T.Compose([T.ToTensor(), T.Normalize(means, stds)])

def get_mean_std():
    with open('mean_and_std', 'rb') as file:
        raw = file.read()
        values = pickle.loads(raw)

    return values[0:3], values[3:6]


class ReplayBuffer(object):
	def __init__(self, size):
		self.size = size
		self.memory = []
		self.position = 0


	def push(self, *args):
		if len(self.memory) < self.size:
			self.memory.append(None)
		self.memory[self.position] = Transition(*args)
		# If replay buffer is full, we start overwriting the first entries
		self.position = (self.position + 1) % self.size


	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)


	def get(self, index):
		return self.memory[index]


	def __len__(self):
		return len(self.memory)


class CONV3_FC1(nn.Module):
	def __init__(self, h, w, outputs):
		super(CONV3_FC1, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=3)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=3)
		self.bn2 = nn.BatchNorm2d(32)
		# self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=3)
		self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=3)
		# self.bn3 = nn.BatchNorm2d(64)
		self.bn3 = nn.BatchNorm2d(32)

		def conv2d_size_out(size, kernel_size=5, stride=3):
			return (size - (kernel_size - 1) - 1) // stride + 1
		convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
		convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
		lin_input_size = convw * convh * 32
		fc1_output_size = int(np.round(outputs/4))
		self.fc1 = nn.Linear(lin_input_size, outputs)


	def forward(self, x):
		# print(x.size())

		x = F.relu(self.bn1(self.conv1(x)))
		# print(x.size())
		x = F.relu(self.bn2(self.conv2(x)))
		# print(x.size())

		x = F.relu(self.bn3(self.conv3(x)))
		# print(x.size())
		return self.fc1(x.view(x.size(0), -1))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        if inplanes != planes:
        	self.conv3 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1)
        else:
        	self.conv3 = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # if self.downsample is not None:
            # identity = self.downsample(x)

        if self.conv3:
        	identity = self.conv3(identity)

        out += identity
        out = self.relu(out)

        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                 padding=dilation, groups=groups, bias=False, dilation=dilation)


class Perception_Module(nn.Module):
	def __init__(self):
		super(Perception_Module, self).__init__()
		self.C1 = conv3x3(3, 64)
		self.MP1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.RB1 = BasicBlock(64, 128)
		self.MP2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.RB2 = BasicBlock(128, 256)
		self.RB3 = BasicBlock(256, 512)


	def forward(self, x, verbose=0):
		if verbose == 1:
			print('### Perception Module ###')
			print('Input: '.ljust(15), x.size())
		x = self.C1(x)
		if verbose == 1:
			print('After Conv1: '.ljust(15), x.size())
			print(x.max())
		x = self.MP1(x)
		if verbose == 1:
			print('After MP1: '.ljust(15), x.size())
		x = self.RB1(x)
		if verbose == 1:
			print('After RB1: '.ljust(15), x.size())
			print(x.max())
		x = self.MP2(x)
		if verbose == 1:
			print('After MP2: '.ljust(15), x.size())
		x = self.RB2(x)
		if verbose == 1:
			print('After RB2: '.ljust(15), x.size())
		x = self.RB3(x)
		if verbose == 1:
			print('After RB3: '.ljust(15), x.size())

		return(x)


class Grasping_Module(nn.Module):
	def __init__(self):
		super(Grasping_Module, self).__init__()
		self.RB1 = BasicBlock(512, 256)
		self.RB2 = BasicBlock(256, 128)
		self.UP1 = nn.UpsamplingBilinear2d(scale_factor=2)
		self.RB3 = BasicBlock(128, 64)
		self.UP2 = nn.UpsamplingBilinear2d(scale_factor=2)
		self.C1 = nn.Conv2d(64, 1, kernel_size=1)
		self.sigmoid = nn.Sigmoid()


	def forward(self, x, verbose=0):
		if verbose == 1:
			print('### Grasping Module ###')
			print('Input: '.ljust(15), x.size())
		x = self.RB1(x)
		if verbose == 1:
			print('After RB1: '.ljust(15), x.size())
		x = self.RB2(x)
		if verbose == 1:
			print('After RB2: '.ljust(15), x.size())
		x = self.UP1(x)
		if verbose == 1:
			print('After UP1: '.ljust(15), x.size())
		x = self.RB3(x)
		if verbose == 1:
			print('After RB3: '.ljust(15), x.size())
		x = self.UP2(x)
		if verbose == 1:
			print('After UP2: '.ljust(15), x.size())
		x = self.C1(x)
		if verbose == 1:
			print('After C1: '.ljust(15), x.size())

		x.squeeze_()

		return(self.sigmoid(x))


def RESNET(verbose=0):
	return nn.Sequential(Perception_Module(), Grasping_Module())


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params} ({total_params/1000000:.2f}M)")


if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	vision_module = Perception_Module()
	grasping_module = Grasping_Module()
	test_input = torch.ones(5,3,200,200).to(device)

	# old = CONV3_FC1(200, 200, 40000).to(device)

	resnet = RESNET().to(device)

	with torch.no_grad():
		out = resnet(test_input)

	# out = old(test_input)

	count_parameters(resnet)

	time.sleep(10)

	print(out.size())