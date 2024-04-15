import torch
import numpy as np
from model import CVAE

model = torch.load("MNIST_CVAE.pt")
model.eval()

d = np.load("mu_sigam.npz", allow_pickle=True)
mu_min = d["mu_min"].reshape(-1, 10)
mu_max = d["mu_max"].reshape(-1, 10)
sigma_min = d["sigma_min"].reshape(-1, 10)
sigma_max = d["sigma_max"].reshape(-1, 10)

def infer(num, times=1):
	with torch.no_grad():
		mu = np.random.uniform(mu_min, mu_max, size=(times, 10))
		sigma = np.random.uniform(sigma_min, sigma_max, size=(times, 10))
		sigma = torch.exp(0.5 * torch.tensor(sigma))
		z = torch.rand_like(sigma)
		z = torch.tensor(mu) + sigma * z
		z = z.float()
		
		y_encoded = torch.nn.functional.one_hot(torch.tensor([num]*times), 10).reshape(-1, 10)
		out = model.decoder(z, y_encoded)
		out = np.uint8(out * 255)

	return out

if __name__ == '__main__':
	images = infer(0, 10)