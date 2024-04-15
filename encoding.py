import torch
from model import CVAE
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import math
import numpy as np
from tqdm import tqdm

test_data = datasets.MNIST(root="../data", train=False, download=True, transform=ToTensor())
train_data = datasets.MNIST(root="../data", train=True, download=True, transform=ToTensor())

model = torch.load("MNIST_CVAE.pt")
model.eval()

def get_encode():
	mus = []
	sigmas = []
	for x, _ in tqdm(test_data):
		y_encoded = torch.nn.functional.one_hot(torch.tensor([0]), 10).reshape(-1, 10)
		x = x.reshape(-1, 1, 28, 28)
		_, mu, sigma = model(x, y_encoded)
		mus.append(mu.numpy().reshape(10))
		sigmas.append(sigma.numpy().reshape(10))
	for x, _ in tqdm(train_data):
		y_encoded = torch.nn.functional.one_hot(torch.tensor([0]), 10).reshape(-1, 10)
		x = x.reshape(-1, 1, 28, 28)
		_, mu, sigma = model(x, y_encoded)
		mus.append(mu.numpy().reshape(10))
		sigmas.append(sigma.numpy().reshape(10))
	return mus, sigmas


with torch.no_grad():
	mus, sigmas = get_encode()
	mus, sigmas = np.array(mus), np.array(sigmas)
	mu_min = np.min(mus, axis=0)
	mu_max = np.max(mus, axis=0)
	sigma_min = np.min(sigmas, axis=0)
	sigma_max = np.max(sigmas, axis=0)
	np.savez("mu_sigam.npz", mu_min=mu_min, mu_max=mu_max, sigma_min=sigma_min, sigma_max=sigma_max)