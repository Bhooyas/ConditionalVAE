import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import CVAE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
train_data = datasets.MNIST(root="../data", train=True, download=True, transform=ToTensor())
image_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
image_channel = 1
z_dim = 10
classes = 10
lr = 0.001
epochs = 100


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
torch.save(model, "MNIST_CVAE.pt")