from xml.parsers.expat import model

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# 1. Model Parameters (Based on Section 3.1 and SI Appendix)
F, a, b, c = 0.0, 1.0, 0.0, 1.0  
# F, a, b, c = 0.5, 1.0, -0.5, 0.5  # Example coefficients for non-Gaussian behavior
sigma = 0.5
dt = 0.01
num_steps = 100000


# 2. Generate Data via Euler-Maruyama Integration
def generate_scalar_data(F, a, b, c, sigma, dt, steps):
    x = torch.zeros(steps)
    curr_x = torch.tensor(0.0)
    for i in range(steps):
        # Deterministic drift: F(x) = F + ax + bx^2 - cx^3
        drift = F + a*curr_x + b*(curr_x**2) - c*(curr_x**3)
        # Stochastic diffusion
        diffusion = sigma * np.sqrt(dt) * torch.randn(1)
        curr_x = curr_x + drift * dt + diffusion
        x[i] = curr_x
    return x.view(-1, 1)

# data = generate_scalar_data(F, a, b, c, sigma, dt, num_steps)

# 3. KGMM-inspired Score Network 
# The paper suggests KGMM for low-dimensional systems
class ScoreNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 1024),
            nn.ReLU(),
            nn.Tanh(),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )
    def forward(self, x):
        return self.net(x)

def train(data, epochs=3000, batch_size=1024, K=4):
    model = ScoreNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    noise_scale = 0.05

    # # Estimate mean and std from clean training data
    # mu = data.mean()
    # sigma_x = data.std()
    for epoch in range(epochs + 1):
        perm = torch.randperm(data.size(0))
        batch = data[perm[:batch_size]]

        # DSM loss on noisy samples
        eps = torch.randn_like(batch) * noise_scale
        x_noisy = batch + eps

        score_pred = model(x_noisy)
        target = -eps / (noise_scale**2)

        loss_dsm = torch.mean((score_pred - target)**2)

        loss = loss_dsm

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, DSM Loss: {loss_dsm.item():.4f}")
    return model