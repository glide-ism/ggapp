import cupy as cp
import matplotlib.pyplot as plt
from scipy.special import gamma
from pathlib import Path

import torch

from ggapp.model import MaternPrior
from ggapp.torch import GGaPPWhiten, GGaPPMap

L = cp.float32(1.0)
nx = 1024
ny = 1024
dx = L/nx

model = MaternPrior(n_levels=8,ny=ny,nx=nx,dx=dx)
model.mg.parameters.sigma.set(1)
model.mg.parameters.l.set(0.1)
model.mg.parameters.nu.set(1)
model.forward_solver.fas_options.report_norms.set(False)

"""
model.forward_solver.fas_options.report_norms.set(False)

s = model.sample()
u = torch.randn(ny,nx,dtype=torch.float32,device='cuda',requires_grad=True)
u.data[:,:] = 0.0
#u = torch.tensor(s,requires_grad=True)

optimizer = torch.optim.SGD([u],lr=1e-3)
for i in range(250):
    optimizer.zero_grad()
    J_data = ((u[:,1000] - 1)**2).sum() + ((u[500,:])**2).sum()
   
    z = GGaPPWhiten.apply(model,u)
    J_prior = (z**2).sum()

    J = J_data + J_prior
    J.backward()

    g_data = u.grad
    Lg_data = GGaPPMap.apply(model,g_data)
    Sigmag_data = GGaPPMap.apply(model,Lg_data)

    u.grad[:,:] = Sigmag_data
 
    optimizer.step()
    print(J.item(), J_data.item(), J_prior.item())
"""
