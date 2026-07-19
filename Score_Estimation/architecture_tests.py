from xml.parsers.expat import model

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import dsm_with_stein
# 1. Model Parameters (Based on Section 3.1 and SI Appendix)
F, a, b, c = 0.0, 1.0, 0.0, 1.0
sigma = 0.5 
dt = 0.01
num_steps = 100000

data = dsm_with_stein.generate_scalar_data(F, a, b, c, sigma, dt, num_steps)
losses = []

for i in range(1):
    model, curr_loss = dsm_with_stein.train(data)
    losses.append(curr_loss)

print("Results:", losses)
print("Mean:", np.mean(losses))
print("Std:", np.std(losses))