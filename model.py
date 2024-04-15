import torch
import torch.nn as nn

class CVAE(nn.Module):

	def __init__(self, image_channel=1, z_dim=20, classes=10):
		super(CVAE, self).__init__()

		self.tanh = nn.Tanh()
		self.relu = nn.ReLU()

		self.conv1 = nn.Conv2d(image_channel, 16, 5, stride=2)
		self.conv2 = nn.Conv2d(16, 32, 5, stride=2)

		self.mu_layer = nn.Linear(512, z_dim)
		self.sigma_layer = nn.Linear(512, z_dim)

		# self.flatten = lambda x: x.reshape(x.shape[0], -1)

		self.d_lin = nn.Linear(z_dim + classes, 512)

		# self.unflatten = lambda x: x.reshape(x.shape[0], 32, 4, 4)

		self.conv_t1 = nn.ConvTranspose2d(32, 16, 5, stride=2)
		self.conv_t2 = nn.ConvTranspose2d(16, image_channel, 5, stride=2)
		self.conv_t3 = nn.ConvTranspose2d(image_channel, image_channel, 4)

	def flatten(self, x):
		return x.reshape(x.shape[0], -1)

	def unflatten(self, x):
		return x.reshape(x.shape[0], 32, 4, 4)

	def encoder(self, x):
		x = self.tanh(self.conv1(x))
		x = self.tanh(self.conv2(x))
		x = self.flatten(x)
		mu = self.tanh(self.mu_layer(x))
		sigma = torch.sigmoid(self.sigma_layer(x))
		return mu, sigma

	def decoder(self, z, y):
		z = torch.cat((z, y), dim=1)
		z = self.tanh(self.d_lin(z))
		z = self.unflatten(z)
		z = self.tanh(self.conv_t1(z))
		z = self.tanh(self.conv_t2(z))
		z = torch.sigmoid(self.conv_t3(z))
		return z

	def forward(self, x, y):
		mu, sigma = self.encoder(x)
		sigma = torch.exp(0.5 * sigma)
		z = torch.rand_like(sigma)
		z = mu + sigma * z
		out = self.decoder(z, y)
		return out,mu,sigma