# # python -m pip install torch
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# import numpy as np

# # 1. THE MODEL
# class ScoreNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(2, 128), nn.ReLU(),
#             nn.Linear(128, 128), nn.ReLU(),
#             nn.Linear(128, 2)
#         )
#     def forward(self, x):
#         return self.net(x)

# # 2. THE DATA (A simple ring shape)
# def get_ring_data(n=2000):
#     theta = torch.rand(n) * 2 * np.pi
#     r = 2.0 + torch.randn(n) * 0.1  # Radius of 2 with slight noise
#     return torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)

# # 3. TRAINING FUNCTION
# def train(model, data, epochs=3000, sigma=0.1):
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)
#     print("Training...")
#     for epoch in range(epochs):
#         # Sample batch
#         idx = torch.randint(0, len(data), (128,))
#         x = data[idx]
        
#         # DSM: Corrupt data and predict score
#         noise = torch.randn_like(x)
#         tilde_x = x + sigma * noise
#         target = -(tilde_x - x) / (sigma**2)
        
#         loss = 0.5 * ((model(tilde_x) - target)**2).sum(dim=-1).mean()
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         if epoch % 500 == 0:
#             print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# # 4. SAMPLING FUNCTION
# def sample(model, steps=500, eps=0.001):
#     model.eval()
#     # Start from random noise far away
#     x = torch.randn(1000, 2) * 3 
#     with torch.no_grad():
#         for _ in range(steps):
#             score = model(x)
#             noise = torch.randn_like(x)
#             # Langevin update
#             x = x + eps * score + torch.sqrt(torch.tensor(2 * eps)) * noise
#     return x

# # --- EXECUTION ---
# dataset = get_ring_data()
# score_model = ScoreNet()

# train(score_model, dataset)
# generated_samples = sample(score_model)

# # Visualize results
# plt.scatter(dataset[:,0], dataset[:,1], alpha=0.3, label="Real Data")
# plt.scatter(generated_samples[:,0], generated_samples[:,1], alpha=0.3, color='red', label="Generated")
# plt.legend()
# plt.show()
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt

# # 1. Generate Synthetic Scalar Data (Two clusters at -3 and 3)
# def get_data(n=2000):
#     return torch.cat([torch.randn(n//2, 1) - 3.0, torch.randn(n//2, 1) + 3.0])

# # 2. Define a simple Score Network
# class ScoreNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(1, 64),
#             nn.Tanh(), # Tanh or Softplus work well for smooth gradients
#             nn.Linear(64, 64),
#             nn.Tanh(),
#             nn.Linear(64, 1)
#         )
#     def forward(self, x):
#         return self.net(x)

# # 3. Training Setup
# data = get_data()
# model = ScoreNet()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# sigma = 0.5  # Fixed noise scale for denoising

# print("Training Score Model...")
# for epoch in range(3001):
#     # Denoising Score Matching Objective:
#     # 1. Add noise to the data
#     eps = torch.randn_like(data) * sigma
#     x_noisy = data + eps
    
#     # 2. Predict the score
#     score_pred = model(x_noisy)
    
#     # 3. Target is -(x_noisy - x) / sigma^2, which simplifies to -eps / sigma^2
#     target = -eps / (sigma**2)
    
#     loss = torch.mean((score_pred - target)**2)
    
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
    
#     if epoch % 1000 == 0:
#         print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# # 4. Sampling using the learned Score (Langevin Dynamics)
# # We start with pure noise and follow the score "uphill"
# x_samples = torch.linspace(-10, 10, 50).view(-1, 1) # Range for visualization
# with torch.no_grad():
#     learned_scores = model(x_samples)

# # Simple Plotting
# plt.figure(figsize=(10, 5))
# plt.hist(data.numpy(), bins=50, density=True, alpha=0.3, label='Original Data')
# plt.plot(x_samples.numpy(), learned_scores.numpy(), color='red', label='Estimated Score $s(x)$')
# plt.axhline(0, color='black', linestyle='--')
# plt.title("Estimated Score Function vs Data Distribution")
# plt.legend()
# plt.show()
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

data = generate_scalar_data(F, a, b, c, sigma, dt, num_steps)

# 3. KGMM-inspired Score Network 
# The paper suggests KGMM for low-dimensional systems
class ScoreNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Tanh(),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.net(x)

model = ScoreNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
noise_scale = 0.05 # G parameter mentioned in paper

# 4. Training Loop (Denoising Score Matching)
print("Training score estimator...")
for epoch in range(2001):
    perm = torch.randperm(data.size(0))
    # Using a batch to speed up training
    batch = data[perm[:1024]]
    
    eps = torch.randn_like(batch) * noise_scale
    x_noisy = batch + eps
    
    score_pred = model(x_noisy)
    target = -eps / (noise_scale**2) # Ground truth score for Gaussian noise
    
    loss = torch.mean((score_pred - target)**2)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# 5. Validation: Compare Analytic vs. Learned Score
x_range = torch.linspace(data.min(), data.max(), 100).view(-1, 1)
with torch.no_grad():
    learned_score = model(x_range)

# Analytic score from the paper: s(x) = (2/sigma^2) * F(x)
analytic_score = (2 / (sigma**2)) * (F + a*x_range + b*(x_range**2) - c*(x_range**3))

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(x_range.numpy(), analytic_score.numpy(), 'b-', label='Analytic Score')
plt.plot(x_range.numpy(), learned_score.numpy(), 'r--', label='Learned Score (ScoreNet)')
plt.axhline(0, color='black', lw=0.5)
plt.title("Comparison of Analytic vs. Learned Score Functions")
plt.xlabel("State x")
plt.ylabel("Score s(x)")
plt.legend()
# plt.show()

def calculate_response(data_series, score_func, max_lag=500):
    """
    Computes R(t) = <x(t) * B(x(0))> where B(x) = -score(x)
    """
    n = len(data_series)
    with torch.no_grad():
        # 1. Compute B(x) = -score(x) for the entire trajectory
        # We negate the score because B(x) = -d/dx log rho
        b_values = -score_func(data_series).flatten()
        
    # 2. Prepare observable A(x) = x (normalized to zero mean)
    a_values = data_series.flatten() - data_series.mean()
    
    # 3. Calculate correlation for different time lags
    lags = np.arange(max_lag)
    response = []
    
    for lag in lags:
        # Correlation: Average of A(t+lag) * B(t)
        if lag == 0:
            corr = torch.mean(a_values * b_values)
        else:
            corr = torch.mean(a_values[lag:] * b_values[:-lag])
        response.append(corr.item())
        
    return lags, np.array(response)

# --- Execution ---

# 1. Get Learned Response
lags, learned_R = calculate_response(data, model)

# 2. Get Analytical Response (using the analytic score formula)
def analytic_score_fn(x):
    return (2 / (sigma**2)) * (F + a*x + b*(x**2) - c*(x**3))

lags, analytic_R = calculate_response(data, analytic_score_fn)

# 3. Plotting
plt.figure(figsize=(10, 5))
plt.plot(lags * dt, analytic_R, 'b-', label='Analytic GFDT')
plt.plot(lags * dt, learned_R, 'r--', label='Learned (ScoreNet) GFDT')
plt.axhline(0, color='black', lw=0.5)
plt.title("Linear Response Function $R(t)$ for the Mean")
plt.xlabel("Time Lag ($t$)")
plt.ylabel("Response Strength")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# # plot the response function based on the learned score
# plt.figure(figsize=(10, 5))
# plt.plot(x_range.numpy(), F + a*x_range + b*(x_range**2) - c*(x_range**3), 'g-', label='Response Function F(x)')
# plt.plot(x_range.numpy(), learned_score.numpy(), 'r--', label='Learned Response Function')
# plt.axhline(0, color='black', lw=0.5)
# plt.title("Response Function F(x)")
# plt.xlabel("State x")
# plt.ylabel("F(x)")
# plt.legend()
# plt.show()