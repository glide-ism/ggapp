import cupy as cp
import matplotlib.pyplot as plt
from scipy.special import gamma
from pathlib import Path

from ggapp.model import MaternPrior

L = cp.float32(1.0)
nx = 2048
ny = 2048
dx = L/nx

# Instantiate a high-frequency GP
model_fast = MaternPrior(n_levels=8,ny=ny,nx=nx,dx=dx)
model_fast.mg.parameters.nu.set(1)
model_fast.mg.parameters.sigma.set(1)
model_fast.mg.parameters.l.set(0.1)
model_fast.forward_solver.fas_options.report_norms.set(False)

# Instantiate a low-frequency GP
model_slow = MaternPrior(n_levels=8,ny=ny,nx=nx,dx=dx)
model_slow.mg.parameters.nu.set(5)
model_slow.mg.parameters.sigma.set(1)
model_slow.mg.parameters.l.set(0.2)
model_slow.forward_solver.fas_options.report_norms.set(False)

# Plot samples
fig,axs = plt.subplots(ncols=2)
axs[0].imshow(model_fast.sample().get())
axs[1].imshow(model_slow.sample().get())
plt.show()


