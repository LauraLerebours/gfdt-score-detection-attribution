from dsm_with_stein import generate_scalar_data, ScoreNet, stein_loss, train
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
# 2. Generate Data via Euler-Maruyama Integration + Stein Test
data = generate_scalar_data(F, a, b, c, sigma, dt, num_steps)
# Train the score network using the Stein loss
model = train(data, epochs=3000, batch_size=1024, K=4)

# Validation
x_range = torch.linspace(data.min(), data.max(), 100).view(-1, 1)
with torch.no_grad():
    learned_score = model(x_range)

analytic_score = (2 / (sigma**2)) * (F + a*x_range + b*(x_range**2) - c*(x_range**3))
plt.figure(figsize=(10, 5))
plt.plot(x_range.numpy(), analytic_score.numpy(), 'b-', label='Analytic Score')
plt.plot(x_range.numpy(), learned_score.numpy(), 'r--', label='Learned Score (ScoreNet)')
plt.axhline(0, color='black', lw=0.5)
plt.title("Comparison of Analytic vs. Learned Score Functions")
plt.xlabel("State x")
plt.ylabel("Score s(x)")
plt.legend()
plt.show()
