import cupy as cp
#from .grid import MaternGrid
from .multigrid import Multigrid, FASSolver

class MaternPrior:
    def __init__(self,mg=None,
            n_levels=None,grid=None,
            ny=None,nx=None,dx=None,
            x0=cp.float32(0.0),y0=cp.float32(0.0),crs=None):
        if mg is not None:
            self.mg = mg
        elif grid is not None and n_levels is not None:
            self.mg = Multigrid(n_levels,finest_grid=grid)
        elif ny and nx and dx and n_levels:
            self.mg = Multigrid(n_levels,ny=ny,nx=nx,dx=dx,
                   x0=x0,y0=y0,crs=crs)
        else:
            raise ValueError('Must supply either (a) a multigrid object \
                              (b) a grid and number of levels \
                              (c) ny/nx/dx and number of levels')

        self._forward_solver = None
        self._adjoint_solver = None
        self.top_level = 0

    @property
    def forward_solver(self):
        if self._forward_solver is None:
            self._forward_solver = FASSolver(self.mg)
        return self._forward_solver
    
    def set_top_level(self,level):
        self.top_level = level

    def solve(self,rhs):
        self.mg.forcing.f.set(rhs,start_level=self.top_level)
        self.forward_solver.solve(start_level=self.top_level)

    def sample(self):
        eta = cp.random.randn(self.mg[self.top_level].ny,self.mg[self.top_level].nx,dtype=cp.float32)
        f = eta
        for i in range(self.mg[self.top_level].parameters.alpha.value//2):
            self.solve(f)
            f = self.mg[self.top_level].state.u.data

        return cp.array(self.mg[self.top_level].state.u.data * self.mg[self.top_level].parameters.tau.value / self.mg[self.top_level].dx)

    def forward(self,x):
        f = x
        for i in range(self.mg[self.top_level].parameters.alpha.value//2):
            self.solve(f)
            f = self.mg[self.top_level].state.u.data

        return cp.array(self.mg[self.top_level].state.u.data * self.mg[self.top_level].parameters.tau.value / self.mg[self.top_level].dx)

    def whiten(self,x):
        for i in range(self.mg[self.top_level].parameters.alpha.value//2):
            self.mg[self.top_level].state.u.set(x)
            self.mg[self.top_level].forward_operators.compute_residual(operator_only=True)
            x = self.mg[self.top_level].forward_operators.F_u
        return cp.array(x / self.mg[self.top_level].parameters.tau.value * self.mg[self.top_level].dx)

    def log_probability(self,x):
        self.mg[self.top_level].state.u.set( self.mg[self.top_level].parameters.tau.value * x )
        self.mg[self.top_level].forward_operators.compute_residual(operator_only=True)
        return cp.linalg.norm(self.mg[self.top_level].forward_operators.F_u)

        



            
