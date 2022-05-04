"""
	This file contains a neural network module for us to
	define our actor and critic networks in PPO.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class FeedForwardNN(nn.Module):
	"""
		A standard in_dim-64-64-out_dim Feed Forward Neural Network.
	"""
	def __init__(self, in_dim, out_dim):
		"""
			Initialize the network and set up the layers.
			Parameters:
				in_dim - input dimensions as an int
				out_dim - output dimensions as an int
			Return:
				None
		"""
		self.in_dim = in_dim
		super(FeedForwardNN, self).__init__()
		if in_dim < 12:
			self.layer1 = nn.Linear(in_dim, 64)
			self.layer2 = nn.Linear(64, 128)
			self.layer3 = nn.Linear(128, 256)
			self.layer4 = nn.Linear(256, 64)
			# print("out dim:", out_dim)
			self.layer5 = nn.Linear(64, out_dim)
		else:
			self.layer1 = nn.Linear(in_dim, 512)
			self.layer2 = nn.Linear(512, 1024)
			self.layer3 = nn.Linear(1024, 1024)
			self.layer4 = nn.Linear(1024, 512)
			self.layer5 = nn.Linear(512, 256)
			# print("out dim:", out_dim)
			self.layer6 = nn.Linear(256, out_dim)

	def forward(self, obs):
		"""
			Runs a forward pass on the neural network.
			Parameters:
				obs - observation to pass as input
			Return:
				output - the output of our forward pass
		"""
		# Convert observation to tensor if it's a numpy array
		if isinstance(obs, np.ndarray):
			obs = torch.tensor(obs, dtype=torch.float)

		if self.in_dim < 12:
			activation1 = F.relu(self.layer1(obs))
			activation2 = F.relu(self.layer2(activation1))
			activation3 = F.relu(self.layer3(activation2))
			activation4 = F.relu(self.layer4(activation3))
			output = self.layer5(activation4)

		else:
			activation1 = F.relu(self.layer1(obs))
			activation2 = F.relu(self.layer2(activation1))
			activation3 = F.relu(self.layer3(activation2))
			activation4 = F.relu(self.layer4(activation3))
			activation5 = F.relu(self.layer5(activation4))
			output = self.layer6(activation5)

		return output
