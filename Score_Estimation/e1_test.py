import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import dsm_with_stein

F, a, b, c = 0.0, 1.0, 0.0, 1.0
sigma = 0.5
dt = 0.01
num_steps = 100000

data = dsm_with_stein.generate_scalar_data(F, a, b, c, sigma, dt, num_steps)

e1 = [0, 500, 1000, 1500, 2000, 2500, 3000]
losses = []

for i in range(10):
    e = 3000 # Change the E1 value here
    print(f'Run {i + 1}/10')
    print(f"Training with E1 = {e}")
    model, curr_loss = dsm_with_stein.train(data, E1=e)
    losses.append(curr_loss)

print("Results:", losses)
print("Mean:", np.mean(losses))
print("Std:", np.std(losses))

# print("Results:", losses)
# plt.plot(e1, losses, marker="o")
# plt.xlabel("E1")
# plt.ylabel("Loss")
# plt.title("Loss vs E1")
# plt.show()


    

