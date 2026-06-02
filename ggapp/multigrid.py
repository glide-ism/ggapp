import cupy as cp
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from .grid import MaternGrid
#from .operators import VankaConfig,NewtonConfig
from glide.field import LocalOption,BroadcastOption

class Multigrid:
    def __init__(self,n_levels: int,finest_grid=None,
            ny=None,nx=None,dx=None,
            x0=cp.float32(0.0),y0=cp.float32(0.0),crs=None,
            use_fast_math=True):

        cuda_dir = Path(__file__).parent / "cuda"

        # Concatenate ice kernel files in dependency order
        cuda_files = ['transfer.cu']
        cuda_source = '\n'.join((cuda_dir / f).read_text() for f in cuda_files)
        
        if use_fast_math:
            options=("--use_fast_math",)
        else:
            options=()

        self.kernels = cp.RawModule(code=cuda_source, options=options)

        if finest_grid is not None:
            print("Instantiating multigrid from existing grid")
            self.finest_grid = finest_grid
        else:
            print("Instantiating multigrid from new grid")
            self.finest_grid = MaternGrid(ny,nx,dx,x0=x0,y0=y0,crs=crs)

        self.n_levels = n_levels

        if n_levels is not None:
            self.create_grid_hierarchy(n_levels,restrict_fields=True)

        self.state = MGStateManager(self)
        self.parameters = MGParameterManager(self)
        self.forcing = MGForcingManager(self)

    def create_grid_hierarchy(self,n_levels,restrict_fields=True):
        self.levels = [self.finest_grid]
        for i in range(1,n_levels):
            coarse_grid = self.create_coarse_grid(self.levels[-1],
                restrict_fields=restrict_fields)
            self.levels.append(coarse_grid)
        return self.levels

    def create_coarse_grid(self,parent_grid,restrict_fields=True):
        child_grid = MaternGrid(
            parent_grid.ny // 2, parent_grid.nx // 2,
            parent_grid.dx * 2, 
            x0=parent_grid.x0 + parent_grid.dx/2,
            y0=parent_grid.y0 - parent_grid.dx/2,
            crs=parent_grid.crs,
            parent=parent_grid
        )
        parent_grid.child = child_grid
        if restrict_fields == True:
            self.restrict_state(parent_grid,child_grid)
            self.restrict_parameters(parent_grid,child_grid)
            self.restrict_forcing(parent_grid,child_grid)
        return child_grid

    def restrict_state(self,fine_grid,coarse_grid):
        self.restrict_cell(fine_grid.state.u.data,coarse_grid.state.u.data)

    def restrict_forcing(self,fine_grid,coarse_grid):
        self.restrict_cell(fine_grid.forcing.f.data,coarse_grid.forcing.f.data)
    def restrict_parameters(self,fine_grid,coarse_grid):
        coarse_grid.parameters.kappa.set(fine_grid.parameters.kappa.value)

    def restrict_residual(self,fine_grid,coarse_grid):
        self.restrict_cell(fine_grid.forward_operators.r_u,coarse_grid.forward_operators.r_u,method='sum')
   
    def restrict_cell(self,fine_field,coarse_field=None,method='avg'):
        """Restrict u-velocity (vertical face) field to coarse grid."""
        if method == 'avg':
            kernel = self.kernels.get_function('restrict_cell_avg')
        elif method == 'sum':
            kernel = self.kernels.get_function('restrict_cell_sum')
        else:
            raise TypeError('Valid restriction methods: [avg,sum]')

        ny, nx = fine_field.shape
        ny_coarse = ny // 2
        nx_coarse = nx // 2

        if coarse_field is None:
            coarse_field = cp.empty((ny_coarse, nx_coarse), dtype=cp.float32)

        total_work = ny_coarse * nx_coarse
        block_size = 256
        grid_size = (total_work + block_size - 1) // block_size

        kernel((grid_size,), (block_size,),
               (fine_field, coarse_field, ny_coarse, nx_coarse))

        return coarse_field
    def prolongate_cell(self,coarse_field, fine_field=None, method='injection'):
        """Prolongate cell-centered field to fine grid."""
        if method == 'injection':
            kernel = self.kernels.get_function('prolongate_cell_injection')
        elif method == 'bilinear':
            kernel = self.kernels.get_function('prolongate_cell_bilinear')
        else:
            raise TypeError('Valid prolongation methods: [injection, bilinear]')

        ny, nx = coarse_field.shape
        ny_fine = ny * 2
        nx_fine = nx * 2

        if fine_field is None:
            fine_field = cp.empty((ny_fine, nx_fine), dtype=cp.float32)

        total_work = ny_fine * nx_fine
        block_size = 256
        grid_size = (total_work + block_size - 1) // block_size

        kernel((grid_size,), (block_size,),
               (coarse_field, fine_field, ny_fine, nx_fine))
        return fine_field   

    def __getitem__(self,key):
        return self.levels[key]

class HierarchyFieldManager:
    def __init__(self, levels, getter, restrict,name=None):
        self._levels = levels
        self._getter = getter
        self._restrict = restrict
        self._name = name

    def set(self, value, start_level=0):
        finest = self._getter(self._levels[start_level])
        finest.set(value)
        self.restrict_down(start_level)

    def restrict_down(self, start_level):
        for l in range(start_level,len(self._levels) - 1):
            fine = self._getter(self._levels[l])
            coarse = self._getter(self._levels[l + 1])
            self._restrict(fine, coarse)

    def set_level(self, level, value):
        self._getter(self._levels[level].grid).set(value)

class MGStateManager:
    def __init__(self, mg):
        self.mg = mg
        self.u = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.state.u,
            restrict=lambda f,c: mg.restrict_cell(f.data,c.data),
            name="u",
        )

    def __repr__(self):
        return f'Top-level ({self.mg.n_levels} levels): \n'+self.mg.levels[0].state.__repr__()

class MGForcingManager:
    def __init__(self, mg):
        self.mg = mg
        self.f = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.forcing.f,
            restrict=lambda f,c: mg.restrict_cell(f.data,c.data,method='avg'),
            name="f",
        )

    def __repr__(self):
        return f'Top-level ({self.mg.n_levels} levels): \n'+self.mg.levels[0].forcing.__repr__()

class MGParameterManager:
    def __init__(self, mg):
        self.mg = mg
        self.l = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.parameters.l,
            restrict=lambda f,c: c.set(f.value),
            name="l",
        )

        self.sigma = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.parameters.sigma,
            restrict=lambda f,c: c.set(f.value),
            name="sigma",
        )

        self.nu = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.parameters.nu,
            restrict=lambda f,c: c.set(f.value),
            name="nu",
        )

        self.delta = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.parameters.delta,
            restrict=lambda f,c: c.set(f.value),
            name="delta",
        )

    def __repr__(self):
        return f'Top-level ({self.mg.n_levels} levels): \n'+self.mg.levels[0].forcing.__repr__()

class JacobiOptions:
    """
    Broadcast wrapper around VankaConfig across all solver levels.
    """

    def __init__(self, levels, getter):
        self._levels = levels
        self._getter = getter  # level -> VankaConfig

        self.options = ['omega']

        self.omega = BroadcastOption(self._levels, self._getter, "omega")


    def set(self, **kwargs):
        validate_kwargs(JacobiConfig, kwargs)
        for lev in self._levels:
            cfg = self._getter(lev)
            for k, v in kwargs.items():
                setattr(cfg, k, v)

    def set_level(self, level_index: int, **kwargs):
        validate_kwargs(JacobiConfig, kwargs)
        cfg = self._getter(self._levels[level_index])
        for k, v in kwargs.items():
            setattr(cfg, k, v)

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self.options))

    def __repr__(self):
        return 'jacobi_options={' + ', '.join([f"{getattr(self,o)}" for o in self.options]) + '}'




class FASSolver:
    def __init__(self,multigrid):
        self.multigrid = multigrid
        self.levels = [FASLevel(grid, FASScratch(grid)) for grid in multigrid.levels]

        self._fas_config = FASConfig()
        self.fas_options = FASOptions(self._fas_config)
        
        self.jacobi_options = JacobiOptions(
            self.levels,
            getter=lambda lev: lev.grid.forward_operators.jacobi_config,
        )

        self.n_levels = len(self.levels)
        self.dt = None

    def solve(self, start_level=0, report_norms=True, zero_init=True):
        
        start_level_ = self.multigrid[start_level]

        if zero_init:
            start_level_.state.u.set(0.0)
        
        r0 = start_level_.forward_operators.compute_residual(return_norms=True)
        initial_residual_norm = r0
        relative_residual_norm = cp.float32(1.0)

        if self._fas_config.report_norms:
            print(f"  Initial:   |r0|     = {initial_residual_norm:.2e}")

        absolute_residual_norm = initial_residual_norm
        iteration = 0
        
        while (relative_residual_norm > self._fas_config.relative_tolerance 
                and absolute_residual_norm > self._fas_config.absolute_tolerance
                and iteration < self._fas_config.maximum_vcycles):
            self.vcycle(start_level,finest=True)

            r = start_level_.forward_operators.compute_residual(return_norms=True)

            absolute_residual_norm = r
            relative_residual_norm = absolute_residual_norm / initial_residual_norm
            if self._fas_config.report_norms:
                print(f"  V-cycle {iteration}: |r|/|r0| = {relative_residual_norm:.2e}, ")
            iteration += 1
        if iteration < self._fas_config.maximum_vcycles:
            converged = True
        else:
            converged = False
        return converged

    def vcycle(self, l, finest=False):

        coarse = not finest
        mg = self.multigrid
        dt = self.dt
        level = self.levels[l]

        # Coarsest level - smooth to convergence
        if l == self.n_levels - 1:
            level.grid.forward_operators.jacobi_sweep(self._fas_config.coarsest_steps)
            return

        next_level = self.levels[l+1]

        # Pre-smooth
        level.grid.forward_operators.jacobi_sweep(self._fas_config.pre_steps)

        mg.restrict_cell(level.grid.state.u.data,next_level.grid.state.u.data)

        next_level.scratch.w[:,:] = next_level.grid.state.u.data[:,:]

        # Compute and restrict residual
        level.grid.forward_operators.compute_residual()
        mg.restrict_cell(level.grid.forward_operators.r_u,next_level.grid.forward_operators.r_u,method='avg')
        
        next_level.grid.forward_operators.compute_residual(operator_only=True)
        next_level.grid.forcing.f.data[:,:] = next_level.grid.forward_operators.F_u[:,:] - next_level.grid.forward_operators.r_u[:,:]

        # recursive call
        self.vcycle(l+1)

        # compute coarse_correction
        next_level.scratch.z[:,:] = next_level.grid.state.u.data[:,:] - next_level.scratch.w[:,:]

        mg.prolongate_cell(next_level.scratch.z,level.scratch.z,method='bilinear')

        # Apply fine correction
        level.grid.state.u.data[:,:] += level.scratch.z[:,:]

        # Post-smooth
        level.grid.forward_operators.jacobi_sweep(self._fas_config.post_steps)

class FASScratch:
    def __init__(self,grid):
        ny,nx = grid.ny,grid.nx
        self.w = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)
        self.z = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)

@dataclass
class FASLevel:
    grid: MaternGrid
    scratch: FASScratch

@dataclass
class FASConfig:
    coarsest_steps: int = 200
    pre_steps: int = 5
    post_steps: int = 5
    maximum_vcycles: int = 10
    relative_tolerance: cp.float32 = cp.float32(1e-3)
    absolute_tolerance: cp.float32 = cp.float32(1e-12)
    report_norms: bool = True

class FASOptions:
    """
    User-facing wrapper around a single FASCDConfig.
    """

    def __init__(self, config: FASConfig):
        self._config = config

        self.options = ['coarsest_steps',
            'pre_steps',
            'post_steps',
            'relative_tolerance',
            'absolute_tolerance',
            'maximum_vcycles',
            'report_norms']

        self.coarsest_steps = LocalOption(
            getter=lambda: self._config.coarsest_steps,
            setter=lambda v: setattr(self._config, "coarsest_steps", v),
            name="coarsest_steps",
        )
        self.pre_steps = LocalOption(
            getter=lambda: self._config.pre_steps,
            setter=lambda v: setattr(self._config, "pre_steps", v),
            name="pre_steps",
        )
        self.post_steps = LocalOption(
            getter=lambda: self._config.post_steps,
            setter=lambda v: setattr(self._config, "post_steps", v),
            name="post_steps",
        )

        self.maximum_vcycles = LocalOption(
            getter=lambda: self._config.maximum_vcycles,
            setter=lambda v: setattr(self._config, "maximum_vcycles", v),
            name="maximum_vcyles",
        )
        
        self.relative_tolerance = LocalOption(
            getter=lambda: self._config.relative_tolerance,
            setter=lambda v: setattr(self._config, "relative_tolerance", v),
            name="relative_tolerance",
        )
        
        self.absolute_tolerance = LocalOption(
            getter=lambda: self._config.absolute_tolerance,
            setter=lambda v: setattr(self._config, "absolute_tolerance", v),
            name="absolute_tolerance",
        )

        self.report_norms = LocalOption(
            getter=lambda: self._config.absolute_tolerance,
            setter=lambda v: setattr(self._config, "report_norms", v),
            name="report_norms",
        )

    def set(self, **kwargs):
        validate_kwargs(FASConfig, kwargs)
        for k, v in kwargs.items():
            setattr(self._config, k, v)

    def __repr__(self):
        return 'fas_options={' + ', '.join([f"{getattr(self,o)}" for o in self.options]) + '}'

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self.options))
