import numpy as np
import matplotlib.pyplot as plt
# 1. Define Model Parameters
F, a, b, c = 0.0, 1.0, 0.0, 1.0  
sigma = 1.0                      
dt = 0.001                        
num_steps = 500000               

# 2. Simulate Unperturbed Time-Series (Euler-Maruyama integration)
x = np.zeros(num_steps)
for i in range(num_steps - 1):
    drift = F + a*x[i] + b*(x[i]**2) - c*(x[i]**3)
    diffusion = sigma * np.sqrt(dt) * np.random.randn()
    x[i+1] = x[i] + drift * dt + diffusion

# 3. Define the State-Dependent Perturbation
# Let's use a nonlinear perturbation: u(x) = 0.5 * x^2
# The derivative is u'(x) = x
u_x = 2 * (x**2)
du_dx = 4*x

# 4. Calculate the Conjugate Observable B(x)
# First, calculate the analytic score s(x)
s_x = (2.0 / sigma**2) * (F + a*x + b*(x**2) - c*(x**3))

# Calculate B(x) using the full GFDT formula for state-dependent forcing
B_x = -du_dx - (u_x * s_x)

# 5. Define the Observables A(x) for the central moments
mu = np.mean(x)
A_x_1 = x                 # 1st moment (Mean)
A_x_2 = (x - mu)**2       # 2nd central moment (Variance)
A_x_3 = (x - mu)**3       # 3rd central moment (Skewness)
A_x_4 = (x - mu)**4       # 4th central moment (Kurtosis)

# 6. Calculate Response Functions 
max_lag_steps = 500
time_lags = np.arange(max_lag_steps) * dt

R_t_1 = np.zeros(max_lag_steps)
R_t_2 = np.zeros(max_lag_steps)
R_t_3 = np.zeros(max_lag_steps)
R_t_4 = np.zeros(max_lag_steps)

# Compute time-lagged cross-correlation over the unperturbed ensemble
for lag in range(max_lag_steps):
    R_t_1[lag] = np.mean(A_x_1[lag:] * B_x[:num_steps-lag])
    R_t_2[lag] = np.mean(A_x_2[lag:] * B_x[:num_steps-lag])
    R_t_3[lag] = np.mean(A_x_3[lag:] * B_x[:num_steps-lag])
    R_t_4[lag] = np.mean(A_x_4[lag:] * B_x[:num_steps-lag])

# 7. Plot the Results
fig, axs = plt.subplots(1, 4, figsize=(15, 3.5))

moments = [R_t_1, R_t_2, R_t_3, R_t_4]
titles = ['1st moment', '2nd moment', '3rd moment', '4th moment']

for i in range(4):
    axs[i].plot(time_lags, moments[i], label='State-Dependent Response', color='purple')
    axs[i].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axs[i].set_title(titles[i])
    axs[i].set_xlabel('Time lag')
    if i == 0:
        axs[i].set_ylabel('Response')
    axs[i].legend()

plt.tight_layout()
plt.show()