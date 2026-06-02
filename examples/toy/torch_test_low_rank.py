import cupy as cp
import matplotlib.pyplot as plt
from scipy.special import gamma
from pathlib import Path

import torch

from ggapp.model import MaternPrior
from ggapp.torch import GGaPPWhiten, GGaPPMap

L = cp.float32(1.0)
nx = 512
ny = 512
dx = L/nx

model = MaternPrior(n_levels=8,ny=ny,nx=nx,dx=dx)
model.mg.parameters.nu.set(1)
model.mg.parameters.sigma.set(1)
model.mg.parameters.l.set(0.01)
model.forward_solver.fas_options.report_norms.set(False)

model2 = MaternPrior(n_levels=8,ny=ny,nx=nx,dx=dx)
model2.mg.parameters.nu.set(1)
model2.mg.parameters.sigma.set(1)
model2.mg.parameters.l.set(0.2)
model2.mg.parameters.delta.set(1.0)
model2.forward_solver.fas_options.report_norms.set(False)

b1 = model2.sample()




"""
n_p = 2

H = torch.zeros(ny,nx,device='cuda')
H.data[:,:] = 100.0
H[ny//3:2*ny//3,nx//3:2*ny//3] = 0.0001

n_samples = 500
Y = torch.randn(ny*nx,n_samples)
for q in range(n_p):
    Q,_ = torch.linalg.qr(Y)
    for j in range(n_samples):
        Y[:,j] = GGaPPMap.apply(model,H*GGaPPMap.apply(model,Q[:,j].reshape(ny,nx))).ravel()
    
C = Q.T @ Y
l,v = torch.linalg.eigh(C)
C_inv_half = v @ torch.diag(l**-0.5) @ v.T
F = Y @ C_inv_half
U,s,_ = torch.linalg.svd(F,full_matrices=False)
lam = s**2

V = torch.zeros(ny*nx,n_samples)
for j in range(n_samples):
    V[:,j] = GGaPPMap.apply(model,U[:,j].reshape(ny,nx)).ravel()
"""



