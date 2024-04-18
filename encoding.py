import torch
from model import CVAE
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import math
import numpy as np
from tqdm import tqdm
from config import *
from safetensors.torch import load_model

test_data = datasets.MNIST(root=data_dir, train=False, download=True, transform=ToTensor())
train_data = datasets.MNIST(root=data_dir, train=True, download=True, transform=ToTensor())

model = CVAE(image_channel=image_channel, z_dim=z_dim, classes=classes)
load_model(model, model_name)
model.eval()

def get_encode():
	mus = []
	sigmas = []
	for x, _ in tqdm(test_data):
		y_encoded = torch.nn.functional.one_hot(torch.tensor([0]), classes).reshape(-1, classes)
		x = x.reshape(-1, 1, 28, 28)
		_, mu, sigma = model(x, y_encoded)
		mus.append(mu.numpy().reshape(z_dim))
		sigmas.append(sigma.numpy().reshape(z_dim))
	for x, _ in tqdm(train_data):
		y_encoded = torch.nn.functional.one_hot(torch.tensor([0]), classes).reshape(-1, classes)
		x = x.reshape(-1, 1, 28, 28)
		_, mu, sigma = model(x, y_encoded)
		mus.append(mu.numpy().reshape(z_dim))
		sigmas.append(sigma.numpy().reshape(z_dim))
	return mus, sigmas


with torch.no_grad():
	mus, sigmas = get_encode()
	mus, sigmas = np.array(mus), np.array(sigmas)
	mu_min = np.min(mus, axis=0)
	mu_max = np.max(mus, axis=0)
	sigma_min = np.min(sigmas, axis=0)
	sigma_max = np.max(sigmas, axis=0)
	np.savez("mu_sigam.npz", mu_min=mu_min, mu_max=mu_max, sigma_min=sigma_min, sigma_max=sigma_max)