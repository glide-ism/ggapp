import cupy as cp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

class ForwardOperators:
    def __init__(self,grid,
            use_fast_math=True):

        self.grid = grid

        cuda_dir = Path(__file__).parent / "cuda"

        # Concatenate ice kernel files in dependency order
        cuda_files = ['residuals.cu', 'jacobi.cu']
        cuda_source = '\n'.join((cuda_dir / f).read_text() for f in cuda_files)
        
        if use_fast_math:
            options=("--use_fast_math",)
        else:
            options=()

        self.kernels = cp.RawModule(code=cuda_source, options=options)

        self.r_u = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)
        self.F_u = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)
        self.z_u = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)
        
        self.jacobi_config = JacobiConfig()
    
    @property
    def _kernel_config(self):
        block_size = (16, 16)
        stride = 14
        halo = 1
        grid_size = (self.grid.nx // stride + 1, self.grid.ny // stride + 1)
        return grid_size, block_size, stride, halo

    def compute_residual(self, 
            operator_only=False, 
            return_norms=False):

        kernel = self.kernels.get_function('compute_residual')
        grid_size, block_size, stride, halo = self._kernel_config
  
        grid = self.grid
        state = grid.state
        parameters = grid.parameters
        forcing = grid.forcing

        if operator_only:
            out_u = self.F_u
            f = self.z_u
        else:
            out_u = self.r_u
            f = forcing.f.data

        kernel(grid_size, block_size,
               (out_u,
                state.u.data,  
                f,
                parameters.kappa.value,
                grid.dx,
                grid.ny, grid.nx, 
                stride, halo)) 

        if return_norms:
            return cp.linalg.norm(out_u)

    def jacobi_smooth(self):

        kernel = self.kernels.get_function('jacobi_smooth')
        grid_size, block_size, stride, halo = self._kernel_config

        grid = self.grid
        state = grid.state
        parameters = grid.parameters
        forcing = grid.forcing

        kernel(grid_size, block_size,
               (state.u.data, 
                self.r_u,
                parameters.kappa.value,
                grid.dx,
                grid.ny, grid.nx, stride, halo, 
                self.jacobi_config.omega)
        )

    def jacobi_sweep(self,n_iter,log_norms=False):
        norms = []
        for i in range(n_iter):
            out = self.compute_residual(return_norms=log_norms)
            if log_norms:
                norms.append(out)
            self.jacobi_smooth()
            
        return norms

@dataclass
class JacobiConfig:
    omega: cp.float32 = cp.float32(0.66)

