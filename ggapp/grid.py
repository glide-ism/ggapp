from dataclasses import dataclass, field, fields
import cupy as cp
from cupyx.scipy.special import gamma

from glide.field import Field, Constant, GridEntity
from .operators import ForwardOperators

@dataclass
class State:
    u: Field | None = None

@dataclass
class Forcing:
    f: Field | None = None

@dataclass
class Coefficients:
    """Spatially varying operator coefficients. `shift` is a per-cell
    diagonal added to the kappa^2 mass term (units of kappa^2, restricted by
    averaging like any density coefficient); zero everywhere reproduces the
    plain Matern operator exactly. Used e.g. as sqrt of a data-precision
    density so the multigrid can invert the shifted factor (L + shift) as a
    preconditioner for the conditioned-prior system (Q + D)."""
    shift: Field | None = None

@dataclass
class Parameters:
    sigma: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(1.0),
            name='sigma',
            units='',
            attrs={'long_name':'Matern marginal variance'})
        )
    l: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(1.0),
            name='l',
            units='m',
            attrs={'long_name':'Matern length scale'})
        )
    nu: Constant = field(
        default_factory=lambda: Constant(
            value=cp.int32(1.0),
            name='nu',
            units='',
            attrs={'long_name':"smoothness parameter"})
        )

    delta: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(1.0),
            name='delta',
            units='',
            attrs={'long_name':"oversmoother"})
        )

    @property
    def d(self):
        return Constant(
                value=cp.int32(2),
                name='d',
                units='',
                attrs={'long_name':'spatial dimension'}
                )

    @property
    def kappa(self):
        return Constant(
                value=cp.float32((8.0 * self.nu.value)**0.5 / self.l.value),
                name='kappa',
                units='',
                attrs={'long_name':'SPDE length scale'}
                )

    @property
    def alpha(self):
        return Constant(
                value=cp.int32(self.nu.value + self.d.value/2),
                name='alpha',
                units='',
                attrs={'long_name':'SPDE smoothness'}
                )


    @property
    def tau(self):
        return Constant(
                value = cp.float32(((self.sigma.value**2 * (4.0 * cp.pi) ** (self.d.value / 2) * self.kappa.value ** (2 * self.nu.value) * gamma(self.alpha.value) / gamma(self.nu.value)) ** 0.5).item()),
                name='tau',
                units='',
                attrs={'long_name':'SPDE noise scale'}
                )

class MaternGrid:
    def __init__(self, ny: int, nx: int,
            dx: cp.float32,
            x0:cp.float32=cp.float32(0.0),
            y0:cp.float32=cp.float32(0.0),
            crs=None,
            parent=None
            ):
        self.ny = ny
        self.nx = nx
        self.dx = cp.float32(dx)
        self.x0 = x0
        self.y0 = y0
        self.crs = crs
        self.x_cell = cp.arange(x0,x0+dx*nx,dx)
        self.y_cell = cp.arange(y0,y0+dx*ny,-dx)

        self.state = self._allocate_state()
        self.forcing = self._allocate_forcing()
        self.parameters = self._allocate_parameters()
        self.coefficients = self._allocate_coefficients()

        self._forward_operators = None
        self._backward_operators = None

    @property
    def forward_operators(self):
        if self._forward_operators is None:
            self._forward_operators = ForwardOperators(self)
        return self._forward_operators


    def _allocate_state(self):
        u = Field(
            data = cp.zeros((self.ny,self.nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=self.dx,
            grid=self,
            name='u',
            units='',
            attrs={'long_name':'matern parameter'}
            )
        return State(u=u)


    def _allocate_forcing(self):
        f = Field(
            data = cp.zeros((self.ny,self.nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=self.dx,
            grid=self,
            name='f',
            units='',
            attrs={'long_name':'matern forcing'}
            )
        return Forcing(f=f)

    def _allocate_parameters(self):
        return Parameters()

    def _allocate_coefficients(self):
        shift = Field(
            data = cp.zeros((self.ny,self.nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=self.dx,
            grid=self,
            name='shift',
            units='',
            attrs={'long_name':'diagonal shift added to the kappa^2 mass term'}
            )
        return Coefficients(shift=shift)




