"""PriorCollection: N Matérn priors sharing one multigrid hierarchy.

A MaternPrior's (ny, nx) buffers carry no state between calls: every solve
zero-initializes `state.u` (FASSolver.solve, zero_init=True) or fully
overwrites it (whiten), `forcing.f` is rewritten by every solve, and the
operator scratch (r_u/F_u/z_u) and FAS scratch (w/z) are within-solve
temporaries. Two priors on the same grid therefore differ ONLY in their
per-level scalar hyperparameters (sigma, l, nu, delta — kappa/alpha/tau are
derived properties) and, optionally, the `coefficients.shift` field.

A PriorCollection exploits this: one shared Multigrid + FASSolver + operator
stack serves any number of named members. Each member (re)binds its scalars
onto the shared hierarchy before every entry point — a handful of broadcast
scalar sets, memoized so consecutive calls on the same member (PCG inner
loops, autograd backward) skip even that. Members with a shift coefficient
(e.g. the ConditionedPrior's shifted preconditioner) own one (ny, nx)
hierarchy of shift arrays, pointer-swapped onto the grids at bind time; the
kernels read `shift.data` per launch, so the swap is exact and copy-free.

Results are bit-identical to separate MaternPriors: binding reproduces
exactly the state that fresh construction plus `mg.parameters.*.set(...)`
would have produced.

NOT safe for concurrent solves — member calls must be serialized. The
single-stream PyTorch autograd engine already guarantees this in the
GGaPPMap/GGaPPWhiten/GGaPPCondition use case.
"""
import cupy as cp

from .model import MaternPrior
from .multigrid import Multigrid


class PriorMember:
    """A named Matérn prior living on a PriorCollection's shared multigrid.

    Quacks like MaternPrior (mg / top_level / forward_solver / set_top_level /
    solve / forward / whiten / sample / log_probability), so GGaPPMap,
    GGaPPWhiten, and ConditionedPrior consume it unchanged. Accessing `.mg`
    binds this member's hyperparameters first, so external introspection
    (e.g. reading `member.mg[0].parameters.tau.value`) also sees this
    member's values.
    """

    def __init__(self, collection, name, sigma, l, nu, delta=1.0, shift=None):
        self._collection = collection
        self.name = name
        self.sigma = float(sigma)
        self.l = float(l)
        self.nu = int(nu)
        self.delta = float(delta)
        # Per-level shift arrays owned by this member (None -> the shared
        # all-zero hierarchy). Restricted once here by plain averaging,
        # exactly as Multigrid.restrict_coefficients does at hierarchy
        # creation.
        if shift is None:
            self._shift = None
        else:
            mg = collection.mg
            levels = [cp.array(shift, dtype=cp.float32)]
            for lev in range(len(mg.levels) - 1):
                coarse = cp.zeros((mg.levels[lev + 1].ny,
                                   mg.levels[lev + 1].nx), dtype=cp.float32)
                mg.restrict_cell(levels[-1], coarse, method='avg')
                levels.append(coarse)
            self._shift = levels

    # ------------------------------------------------ MaternPrior interface
    @property
    def mg(self):
        self._collection.bind(self)
        return self._collection.mg

    @property
    def top_level(self):
        return self._collection._prior.top_level

    @property
    def forward_solver(self):
        return self._collection._prior.forward_solver

    def set_top_level(self, level):
        self._collection._prior.set_top_level(level)

    def solve(self, rhs, zero_init=True):
        self._collection.bind(self)
        return self._collection._prior.solve(rhs, zero_init=zero_init)

    def forward(self, x, zero_init=True):
        self._collection.bind(self)
        return self._collection._prior.forward(x, zero_init=zero_init)

    def whiten(self, x):
        self._collection.bind(self)
        return self._collection._prior.whiten(x)

    def sample(self):
        self._collection.bind(self)
        return self._collection._prior.sample()

    def log_probability(self, x):
        self._collection.bind(self)
        return self._collection._prior.log_probability(x)

    def __repr__(self):
        return (f"PriorMember({self.name!r}, sigma={self.sigma}, l={self.l}, "
                f"nu={self.nu}, delta={self.delta}, "
                f"shifted={self._shift is not None})")


class PriorCollection:
    """N Matérn priors sharing ONE multigrid hierarchy, solver, and scratch.

    Replaces N full hierarchies (state/forcing/shift plus the lazily built
    operator and FAS scratch — 8 hierarchy-fields each) with a single set,
    plus one (ny, nx) hierarchy per *shifted* member. See the module
    docstring for why this is exact.
    """

    def __init__(self, n_levels, ny=None, nx=None, dx=None,
                 x0=cp.float32(0.0), y0=cp.float32(0.0), crs=None,
                 finest_grid=None):
        self.mg = Multigrid(n_levels, finest_grid=finest_grid,
                            ny=ny, nx=nx, dx=dx, x0=x0, y0=y0, crs=crs)
        self._prior = MaternPrior(mg=self.mg)
        self._bound = None
        # The all-zero shift arrays allocated by the grids themselves; every
        # unshifted member points the grids back at these.
        self._zero_shift = [g.coefficients.shift.data for g in self.mg.levels]
        self.members = {}

    def add(self, name, sigma, l, nu, delta=1.0, shift=None):
        """Register a prior with the given hyperparameters and return its
        member handle. `shift` (optional, (ny, nx) at the finest level) gives
        the member a private shift-coefficient hierarchy."""
        if name in self.members:
            raise ValueError(f"duplicate prior member {name!r}")
        member = PriorMember(self, name, sigma, l, nu, delta=delta,
                             shift=shift)
        self.members[name] = member
        return member

    def __getitem__(self, name):
        return self.members[name]

    def bind(self, member):
        """Bind `member`'s hyperparameters (and shift arrays) onto the shared
        hierarchy. Memoized: a no-op when `member` is already bound."""
        if self._bound is member:
            return
        p = self.mg.parameters
        p.sigma.set(member.sigma)
        p.l.set(member.l)
        p.nu.set(member.nu)
        p.delta.set(member.delta)
        shift = member._shift if member._shift is not None else self._zero_shift
        for grid, arr in zip(self.mg.levels, shift):
            grid.coefficients.shift.data = arr
        self._bound = member
