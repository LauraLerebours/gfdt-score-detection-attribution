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

def stein_loss(model, clean_batch, mu, sigma_x, K=4):
    z = (clean_batch - mu) / sigma_x
    score = model(clean_batch)
    residuals = []
    # k = 0 constraint: E[s(X)] = 0
    r0 = score.mean()
    residuals.append(r0**2)
    # k >= 1 constraints:
    # E[z^k s(X)] + (k/sigma_x) E[z^(k-1)] = 0
    for k in range(1, K + 1):
        rk = (z**k * score).mean() + (k / sigma_x) * (z**(k - 1)).mean()
        residuals.append(rk**2)
    return sum(residuals), residuals
def train(data, epochs=3000, batch_size=1024, K=4):
    model = ScoreNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    noise_scale = 0.05

    # Estimate mean and std from clean training data
    mu = data.mean()
    sigma_x = data.std()

    # Stein schedule
    E0 = 1000          # DSM only until this epoch
    E1 = 2000          # ramp finishes here
    lambda_max = 0.1   # tune this

    for epoch in range(epochs + 1):
        perm = torch.randperm(data.size(0))
        batch = data[perm[:batch_size]]

        # DSM loss on noisy samples
        eps = torch.randn_like(batch) * noise_scale
        x_noisy = batch + eps

        score_pred = model(x_noisy)
        target = -eps / (noise_scale**2)

        loss_dsm = torch.mean((score_pred - target)**2)

        # Stein loss on clean samples
        loss_stein, residuals = stein_loss(model, batch, mu, sigma_x, K=K)

        # Ramp Stein penalty
        if epoch < E0:
            lambda_stein = 0.0
        elif epoch <= E1:
            lambda_stein = lambda_max * ((epoch - E0) / (E1 - E0))**2
        else:
            lambda_stein = lambda_max

        loss = loss_dsm + lambda_stein * loss_stein

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            residual_vals = [r.item() for r in residuals]
            print(
                f"Epoch {epoch} | "
                f"DSM: {loss_dsm.item():.4f} | "
                f"Stein: {loss_stein.item():.4f} | "
                f"lambda: {lambda_stein:.4f} | "
                f"residuals: {residual_vals}"
            )

    return model
# G parameter mentioned in paper OLDDD
# def train(data, epochs=2000, batch_size=1024):
#     model = ScoreNet()
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     noise_scale = 0.05 
#     for epoch in range(epochs + 1):
#         perm = torch.randperm(data.size(0))
#         batch = data[perm[:batch_size]]
        
#         eps = torch.randn_like(batch) * noise_scale
#         x_noisy = batch + eps
        
#         score_pred = model(x_noisy)
#         target = -eps / (noise_scale**2)
        
#         loss = torch.mean((score_pred - target)**2)
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         if epoch % 500 == 0:
#             print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
#     return model

# # 4. Training Loop (Denoising Score Matching)
# print("Training score estimator...")
# for epoch in range(2001):
#     perm = torch.randperm(data.size(0))
#     # Using a batch to speed up training
#     batch = data[perm[:1024]]
    
#     eps = torch.randn_like(batch) * noise_scale
#     x_noisy = batch + eps
    
#     score_pred = model(x_noisy)
#     target = -eps / (noise_scale**2) # Ground truth score for Gaussian noise
    
#     loss = torch.mean((score_pred - target)**2)
    
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
    
#     if epoch % 500 == 0:
#         print(f"Epoch {epoch} | Loss: {loss.item():.4f}")


# # 5. Validation: Compare Analytic vs. Learned Score
# x_range = torch.linspace(data.min(), data.max(), 100).view(-1, 1)
# with torch.no_grad():
#     learned_score = model(x_range)

# # Analytic score from the paper: s(x) = (2/sigma^2) * F(x)
# analytic_score = (2 / (sigma**2)) * (F + a*x_range + b*(x_range**2) - c*(x_range**3))

# # Plotting
# plt.figure(figsize=(10, 5))
# plt.plot(x_range.numpy(), analytic_score.numpy(), 'b-', label='Analytic Score')
# plt.plot(x_range.numpy(), learned_score.numpy(), 'r--', label='Learned Score (ScoreNet)')
# plt.axhline(0, color='black', lw=0.5)
# plt.title("Comparison of Analytic vs. Learned Score Functions")
# plt.xlabel("State x")
# plt.ylabel("Score s(x)")
# plt.legend()
# # plt.show()

# def calculate_response(data_series, score_func, max_lag=500):
#     """
#     Computes R(t) = <x(t) * B(x(0))> where B(x) = -score(x)
#     """
#     n = len(data_series)
#     with torch.no_grad():
#         # 1. Compute B(x) = -score(x) for the entire trajectory
#         # We negate the score because B(x) = -d/dx log rho
#         b_values = -score_func(data_series).flatten()
        
#     # 2. Prepare observable A(x) = x (normalized to zero mean)
#     a_values = data_series.flatten() - data_series.mean()
    
#     # 3. Calculate correlation for different time lags
#     lags = np.arange(max_lag)
#     response = []
    
#     for lag in lags:
#         # Correlation: Average of A(t+lag) * B(t)
#         if lag == 0:
#             corr = torch.mean(a_values * b_values)
#         else:
#             corr = torch.mean(a_values[lag:] * b_values[:-lag])
#         response.append(corr.item())
        
#     return lags, np.array(response)

# # --- Execution ---

# # 1. Get Learned Response
# lags, learned_R = calculate_response(data, model)

# # 2. Get Analytical Response (using the analytic score formula)
# def analytic_score_fn(x):
#     return (2 / (sigma**2)) * (F + a*x + b*(x**2) - c*(x**3))

# lags, analytic_R = calculate_response(data, analytic_score_fn)

# # 3. Plotting
# plt.figure(figsize=(10, 5))
# plt.plot(lags * dt, analytic_R, 'b-', label='Analytic GFDT')
# plt.plot(lags * dt, learned_R, 'r--', label='Learned (ScoreNet) GFDT')
# plt.axhline(0, color='black', lw=0.5)
# plt.title("Linear Response Function $R(t)$ for the Mean")
# plt.xlabel("Time Lag ($t$)")
# plt.ylabel("Response Strength")
# plt.legend()
# plt.grid(alpha=0.3)
# plt.show()

# # # plot the response function based on the learned score
# # plt.figure(figsize=(10, 5))
# # plt.plot(x_range.numpy(), F + a*x_range + b*(x_range**2) - c*(x_range**3), 'g-', label='Response Function F(x)')
# # plt.plot(x_range.numpy(), learned_score.numpy(), 'r--', label='Learned Response Function')
# # plt.axhline(0, color='black', lw=0.5)
# # plt.title("Response Function F(x)")
# # plt.xlabel("State x")
# # plt.ylabel("F(x)")
# # plt.legend()
# # plt.show()