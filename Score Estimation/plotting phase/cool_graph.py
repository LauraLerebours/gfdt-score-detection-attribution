import numpy as np
import matplotlib.pyplot as plt
# 1. Define Model Parameters
F, a, b, c = 0.0, 1.0, 0.0, 1.0  
sigma = 1.0                      
dt = 0.01                        
num_steps = 50000    
# rand_1 =np.random.RandomState(seed=42)  # For reproducibility
# rand_start = np.random.RandomState(seed=123)             

# 2. Simulate Unperturbed Time-Series (Euler-Maruyama integration)
def simulate_time_series(F, a, b, c, sigma, dt, num_steps, x_0):
    x = np.zeros(num_steps)
    x[0] = x_0
    for i in range(num_steps - 1):
        drift = F + a*x[i] + b*(x[i]**2) - c*(x[i]**3)
        diffusion = sigma * np.sqrt(dt) * np.random.randn()
        x[i+1] = x[i] + drift * dt + diffusion
    return x


base_x = simulate_time_series(F, a, b, c, sigma, dt, num_steps, x_0=0.0) #Used as true pdf for sampling initial conditions
initial_conditions = np.random.randint(low=0, high=len(base_x)-1, size=10)  # Generate 200 random initial conditions

for n in range(1,10):
    print(f'Simulating for n={n} time steps...')
    avg = []
    for initial in initial_conditions:
        x = simulate_time_series(F, a, b, c, sigma, dt, num_steps, x_0=initial) 
        x_1 = simulate_time_series(F, a, b, c, sigma, dt, num_steps, x_0=initial + 0.1)
        diff = x - x_1
        avg.append(diff)

    plt.plot(avg, color='red')

plt.title('Simulated Average Trajectory Differences for 200 Random Initial Conditions')
plt.xlabel('Time Steps')
plt.ylabel('State Variable x')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()