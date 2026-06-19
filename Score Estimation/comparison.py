import dsm_no_stein
import dsm_with_stein
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
# 1. Model Parameters (Based on Section 3.1 and SI Appendix)
F, a, b, c = 0.0, 1.0, 0.0, 1.0  
# F, a, b, c = 0.5, 1.0, -0.5, 0.5  # Example coefficients for non-Gaussian behavior
sigma = 0.5
dt = 0.01
num_steps = 100000

data = dsm_no_stein.generate_scalar_data(F, a, b, c, sigma, dt, num_steps)
mu = data.mean()
sigma_x = data.std()
K = 4

# count = 0
# for run in range(50):
#     print(f'run: {run}')
#     with_stein_model = dsm_with_stein.train(data, epochs=3000, batch_size=1024, K=K)
#     dsm_only_model = dsm_no_stein.train(data, epochs=3000, batch_size=1024, K=K)
#     with torch.no_grad():
#         stein_old, _ = dsm_with_stein.stein_loss(dsm_only_model, data, mu, sigma_x, K=K)
#         stein_new, _ = dsm_with_stein.stein_loss(with_stein_model, data, mu, sigma_x, K=K)
#     if stein_new < stein_old:
#         count+=1
# print(f"Stein model is better {count}% of the time")
with_stein_model,perm = dsm_with_stein.train(data, epochs=3000, batch_size=1024, K=K)
dsm_only_model = dsm_no_stein.train(data, epochs=3000, batch_size=1024, K=K)
with torch.no_grad():
    stein_old, residuals_old = dsm_with_stein.stein_loss(dsm_only_model, data[perm[:1024]], mu, sigma_x, K=K)
    stein_new, residuals_new = dsm_with_stein.stein_loss(with_stein_model, data[perm[:1024]], mu, sigma_x, K=K)

print("DSM only Stein loss:", stein_old.item())
print("DSM + Stein Stein loss:", stein_new.item())

print("DSM only residuals:", [r.item() for r in residuals_old])
print("DSM + Stein residuals:", [r.item() for r in residuals_new])