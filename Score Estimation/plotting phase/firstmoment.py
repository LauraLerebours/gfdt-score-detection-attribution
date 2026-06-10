import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import time
start_time = time.time()


F, a, b, c = 0.0, 1.0, 0.0, 1.0  
sigma = 1.0                      
dt = 0.01                        
num_steps = 500000      
num_steps_R = 1000         
######## FIRST MOMENT #########
 # For reproducibility
rand_start = np.random.RandomState(seed=123)  # For reproducibility of initial conditions
   
# function to simulate time series with given initial condition
def simulate_time_series(F, a, b, c, sigma, dt, num_steps, x_0, seed):
    rand_noise = np.random.RandomState(seed=seed) 
    x = np.zeros(num_steps)
    x[0] = x_0
    for i in range(num_steps - 1):
        drift = F + a*x[i] + b*(x[i]**2) - c*(x[i]**3)
        diffusion = sigma * np.sqrt(dt) * rand_noise.randn()
        x[i+1] = x[i] + drift * dt + diffusion
    return x
# pertubation magnitude
eta = 0.05  
base_x = simulate_time_series(F, a, b, c, sigma, dt, num_steps, x_0=0.0, seed=123) #Used as true pdf for sampling initial conditions
initial_conditions = rand_start.randint(low=0, high=len(base_x)-1, size=400)  # Generate 200 random initial conditions
trajectory = []
unperturbed = []
perturbed = []
# Simulate trajectories for each initial condition and compute the average difference at the final time step
def simulate_and_compute_diff(initial, eta, dt, num_steps, seed):
    i = base_x[initial] # Get the state variable value at the initial condition index
    x_0 = simulate_time_series(F, a, b, c, sigma, dt, num_steps, x_0=i, seed=seed)
    x_1 = simulate_time_series(F, a, b, c, sigma, dt, num_steps, x_0=eta + i, seed=seed)
    unperturbed.append(x_0)
    perturbed.append(x_1)
    diff = [x_1[j] - x_0[j] for j in range(len(x_0))]
    return diff 

# Generate data for each initial condition
rand_seed = np.random.RandomState(seed=456)  # For reproducibility of random seeds for each n
differences = []
j = 0
for initial in initial_conditions:
    print(j)
    j += 1
    seed_call = rand_seed.randint(low=0, high=10000)  # Generate a random seed for this initial condition 
    diff = simulate_and_compute_diff(initial, eta, dt, num_steps_R, seed_call) # Call the function to simulate and compute the difference
    differences.append(diff)  # Append the difference at the final time step

# Averagae Data
for i in range(len(differences[0])):  # Loop over time steps
    avg_diff = np.mean([diff[i] for diff in differences])  # Average over all initial conditions
    trajectory.append(avg_diff)  # Append the average difference for this time step

# Plot everything
end_time = time.time()
print(f"Total simulation time: {end_time - start_time:.2f} seconds")

plt.plot(trajectory, label='Average Trajectory Difference', color='red')
plt.title('Dynamical Estimation of Perturbation Effect')
plt.xlabel('Time Steps')
plt.ylabel('State Variable x')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

