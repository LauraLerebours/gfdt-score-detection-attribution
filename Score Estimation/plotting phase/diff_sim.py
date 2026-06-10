import numpy as np
import matplotlib.pyplot as plt
# 1. Define Model Parameters
F, a, b, c = 0.0, 1.0, 0.0, 1.0  
sigma = 1.0                      
dt = 0.01                        
num_steps = 500000               

# 2. Simulate Unperturbed Time-Series (Euler-Maruyama integration)
def simulate_time_series(F, a, b, c, sigma, dt, num_steps):
    x = np.zeros(num_steps)
    for i in range(num_steps - 1):
        drift = F + a*x[i] + b*(x[i]**2) - c*(x[i]**3)
        diffusion = sigma * np.sqrt(dt) * np.random.randn()
        # x[i+1] = x[i] + drift * dt 
        x[i+1] = x[i] + drift * dt + diffusion
    return x
# x = np.zeros(num_steps)
# for i in range(num_steps - 1):
#     drift = F + a*x[i] + b*(x[i]**2) - c*(x[i]**3)
#     diffusion = sigma * np.sqrt(dt) * np.random.randn()
#     x[i+1] = x[i] + drift * dt + diffusion

for i in range(1, 10):
    x = simulate_time_series(F, a, b, c, sigma, dt, num_steps)
    plt.plot(x, label=f'n={i} steps')
plt.title('Simulated Time Series for Different Time Steps')
plt.xlabel('Time Steps')
plt.ylabel('State Variable x')
plt.legend()    
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()