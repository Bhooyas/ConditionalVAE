import torch
import numpy as np
from model import CVAE
from config import *
from safetensors.torch import load_model
from fuzzywuzzy import process
import re

model = CVAE(image_channel=image_channel, z_dim=z_dim, classes=classes)
load_model(model, model_name)
model.eval()

d = np.load("mu_sigam.npz", allow_pickle=True)
mu_min = d["mu_min"].reshape(-1, z_dim)
mu_max = d["mu_max"].reshape(-1, z_dim)
sigma_min = d["sigma_min"].reshape(-1, z_dim)
sigma_max = d["sigma_max"].reshape(-1, z_dim)
mapping = {
	"zero": 0,
	"one": 1,
	"two": 2,
	"three": 3,
	"four": 4,
	"five": 5,
	"six": 6,
	"seven": 7,
	"eight": 8,
	"nine": 9
}
examples = open("string_input.txt").read().splitlines()

def get_sample():
	return examples[np.random.randint(0, len(examples))]

def get_num(ip):
	num = re.findall(r"\d", ip.lower())
	if num != []:
		return int(num[0])
	best_matches = process.extractOne(ip.lower(), mapping.keys())
	return mapping[best_matches[0]] if best_matches[1] > 59 else None

def infer(num, times=1):
	with torch.no_grad():
		mu = np.random.uniform(mu_min, mu_max, size=(times, z_dim))
		sigma = np.random.uniform(sigma_min, sigma_max, size=(times, z_dim))
		sigma = torch.exp(0.5 * torch.tensor(sigma))
		z = torch.rand_like(sigma)
		z = torch.tensor(mu) + sigma * z
		z = z.float()
		
		y_encoded = torch.nn.functional.one_hot(torch.tensor([num]*times), classes).reshape(-1, classes)
		out = model.decoder(z, y_encoded)
		out = np.uint8(out * 255)

	return out

if __name__ == '__main__':
	images = infer(0, 10)
	print(get_num(get_sample()))