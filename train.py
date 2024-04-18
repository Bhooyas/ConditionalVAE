import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm
from model import CVAE
from config import *
from safetensors.torch import save_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data = datasets.MNIST(root=data_dir, train=True, download=True, transform=ToTensor())
image_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)


model = CVAE(image_channel=image_channel, z_dim=z_dim, classes=classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
mean = lambda x: sum(x)/len(x)


for epoch in range(epochs):
	pbar = tqdm(image_loader)
	pbar.set_description(f"Epoch {epoch+1}/{epochs}")
	losses = []
	for x,y in pbar:
		x = x.to(device)
		y = torch.nn.functional.one_hot(y, classes).to(device)
		out, mu, sigma = model(x, y)

		out_loss = nn.functional.mse_loss(out, x, reduction="sum")
		kl_div = -0.5 * torch.mean(1 + sigma - mu.pow(2) - sigma.exp())

		loss = 0.0003 * out_loss + kl_div
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		losses.append(loss.item())
		pbar.set_postfix({"Loss": mean(losses)})

model.cpu()
save_model(model, model_name)