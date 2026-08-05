"""Tests for GP conditioning (ggapp/conditioning.py, GGaPPCondition).

Run with:  python -m pytest tests/test_conditioning.py
Requires a CUDA device (cupy); torch tests are skipped without torch.

The grid is tiny (32x32) so the dense kriging reference is exact: Q is built
column-by-column from the same stencil applies the PCG uses, so the reference
isolates the *solver*, not the multigrid preconditioner's accuracy.
"""
import warnings

import numpy as np

try:
    import pytest
except ImportError:                       # allow running without pytest installed
    pytest = None

import cupy as cp
assert cp.cuda.runtime.getDeviceCount() >= 1, "no CUDA device"

try:
    import torch
    _HAVE_TORCH = torch.cuda.is_available()
except ImportError:
    torch = None
    _HAVE_TORCH = False

from ggapp.model import MaternPrior
from ggapp.multigrid import Multigrid
from ggapp.conditioning import ConditionedPrior

if _HAVE_TORCH:
    from ggapp.torch import GGaPPCondition

NY = NX = 32
DX = 1.0
N_LEVELS = 3
SIGMA = 2.0
ELL = 4.0
NU = 1


def make_prior(sigma=SIGMA, l=ELL, nu=NU):
    mg = Multigrid(N_LEVELS, ny=NY, nx=NX, dx=DX, use_fast_math=False)
    prior = MaternPrior(mg=mg)
    prior.mg.parameters.sigma.set(sigma)
    prior.mg.parameters.l.set(l)
    prior.mg.parameters.nu.set(nu)
    prior.forward_solver.fas_options.report_norms.set(False)
    # Tight multigrid tolerance so solver slop stays below test tolerances.
    prior.forward_solver.fas_options.relative_tolerance.set(1e-6)
    prior.forward_solver.fas_options.maximum_vcycles.set(50)
    return prior


def make_data(rng):
    """A border band of 'DEM' cells plus a few interior 'picks'."""
    precision = cp.zeros((NY, NX), dtype=cp.float32)
    precision[:2, :] = precision[-2:, :] = 1.0 / 0.2 ** 2
    precision[:, :2] = precision[:, -2:] = 1.0 / 0.2 ** 2
    picks = [(10, 12), (11, 12), (20, 25), (21, 8), (5, 20)]
    for (i, j) in picks:
        precision[i, j] += 1.0 / 0.5 ** 2
    b = cp.asarray(rng.standard_normal((NY, NX)).astype(np.float32)) * SIGMA
    b *= (precision > 0)
    return b, precision, picks


def make_conditioner(prior=None, rtol=1e-6, seed=0, **kwargs):
    prior = prior or make_prior()
    rng = np.random.default_rng(seed)
    b, precision, picks = make_data(rng)
    c = ConditionedPrior(prior, b, precision, rtol=rtol, maxiter=2000, **kwargs)
    return c, b, precision, picks, rng


def dense_Q(c):
    """Build Q densely from the same stencil applies PCG uses."""
    n = NY * NX
    Q = np.zeros((n, n), dtype=np.float64)
    e = cp.zeros((NY, NX), dtype=cp.float32)
    for k in range(n):
        e.ravel()[k] = 1.0
        Q[:, k] = cp.asnumpy(c.apply_Q(e)).ravel()
        e.ravel()[k] = 0.0
    return 0.5 * (Q + Q.T)   # symmetrize away float32 stencil noise


def dense_solution(c, rhs):
    Q = dense_Q(c)
    A = Q + np.diag(cp.asnumpy(c.precision).ravel().astype(np.float64))
    x = np.linalg.solve(A, cp.asnumpy(rhs).ravel().astype(np.float64))
    return x.reshape(NY, NX)


def rel_err(a, b):
    a = cp.asnumpy(a) if isinstance(a, cp.ndarray) else np.asarray(a)
    b = cp.asnumpy(b) if isinstance(b, cp.ndarray) else np.asarray(b)
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)


# --------------------------------------------------------------------------- #

def test_pcg_matches_dense_kriging():
    c, b, precision, picks, rng = make_conditioner(warm_start=False)
    u0 = cp.asarray(rng.standard_normal((NY, NX)).astype(np.float32)) * SIGMA
    rhs = precision * (b - u0)
    mu_ref = dense_solution(c, rhs)
    mu = c.correct(u0)
    assert c.last_converged
    assert rel_err(mu, mu_ref) < 1e-3


def test_zero_u0_gives_kriging_mean():
    c, b, precision, _, _ = make_conditioner(warm_start=False)
    mean_ref = dense_solution(c, precision * b)
    bed = c.condition(cp.zeros((NY, NX), dtype=cp.float32))
    assert rel_err(bed, mean_ref) < 1e-3


def test_C_is_Q_inverse():
    c, _, _, _, rng = make_conditioner()
    v = cp.asarray(rng.standard_normal((NY, NX)).astype(np.float32))
    assert rel_err(c.apply_Q(c.apply_C(v)), v) < 5e-2


def test_adjoint_identity():
    c, _, _, _, rng = make_conditioner(warm_start=False)
    u = cp.asarray(rng.standard_normal((NY, NX)).astype(np.float32))
    v = cp.asarray(rng.standard_normal((NY, NX)).astype(np.float32))
    Sv = v - c.solve(c.precision * v)[0]
    STu = u - c.precision * c.solve(u)[0]
    lhs = float(cp.sum(STu * v, dtype=cp.float64))
    rhs = float(cp.sum(u * Sv, dtype=cp.float64))
    assert abs(lhs - rhs) / max(abs(rhs), 1e-30) < 1e-3


def test_latent_from_field_round_trip():
    c, _, _, _, rng = make_conditioner(warm_start=False)
    u0 = cp.asarray(rng.standard_normal((NY, NX)).astype(np.float32)) * SIGMA
    bed = c.condition(u0)
    u0_rec = c.latent_from_field(bed, cp.zeros_like(bed))
    assert rel_err(u0_rec, u0) < 5e-2


def test_interpolation_limit():
    prior = make_prior()
    rng = np.random.default_rng(0)
    precision = cp.zeros((NY, NX), dtype=cp.float32)
    picks = [(10, 12), (20, 25), (21, 8)]
    for (i, j) in picks:
        precision[i, j] = 1.0 / 1e-2 ** 2
    b = cp.asarray(rng.standard_normal((NY, NX)).astype(np.float32)) * SIGMA
    c = ConditionedPrior(prior, b, precision, rtol=1e-7, maxiter=5000)
    u0 = cp.asarray(rng.standard_normal((NY, NX)).astype(np.float32)) * SIGMA
    bed = c.condition(u0)
    for (i, j) in picks:
        assert abs(float(bed[i, j] - b[i, j])) < 0.05 * SIGMA


def test_empirical_variance():
    c, b, precision, picks, _ = make_conditioner(rtol=1e-4)
    cp.random.seed(0)
    samples = cp.stack([c.condition(c.prior.sample()) for _ in range(200)])
    sd = cp.asnumpy(samples.std(axis=0))
    # Collapsed at strongly observed cells (border sigma_obs = 0.2)...
    assert sd[:2, :].mean() < 0.35
    # ...and ~sigma_prior far from all data (center is > 3*l from the border).
    center_sd = sd[NY // 2 - 2:NY // 2 + 2, NX // 2 - 2:NX // 2 + 2].mean()
    assert abs(center_sd - SIGMA) / SIGMA < 0.3


def test_shifted_vs_prior_same_solution_fewer_iters():
    """The preconditioner changes the path, never the solution — and the
    shifted factor should need far fewer iterations at this sigma ratio."""
    prior = make_prior()
    rng = np.random.default_rng(0)
    b, precision, _ = make_data(rng)
    u0 = cp.asarray(rng.standard_normal((NY, NX)).astype(np.float32)) * SIGMA
    c_prior = ConditionedPrior(prior, b, precision, rtol=1e-6, maxiter=2000,
                               warm_start=False, preconditioner="prior")
    c_shift = ConditionedPrior(prior, b, precision, rtol=1e-6, maxiter=2000,
                               warm_start=False, preconditioner="shifted")
    mu_ref = dense_solution(c_prior, precision * (b - u0))
    mu_p = c_prior.correct(u0)
    iters_p = c_prior.last_iters
    mu_s = c_shift.correct(u0)
    iters_s = c_shift.last_iters
    assert rel_err(mu_p, mu_ref) < 1e-3
    assert rel_err(mu_s, mu_ref) < 1e-3
    assert c_shift.last_converged
    assert iters_s < iters_p / 2, (iters_s, iters_p)
    assert iters_s <= 40, iters_s


def test_shifted_spectrum_bound():
    """Dense eigenvalues of M^-1 (Q+D) on the small grid: the shifted
    preconditioner should leave only an O(1) spread (the cross term)."""
    c, b, precision, _, _ = make_conditioner(warm_start=False)
    assert c._shifted is not None
    n = NY * NX
    MA = np.zeros((n, n), dtype=np.float64)
    e = cp.zeros((NY, NX), dtype=cp.float32)
    for k in range(n):
        e.ravel()[k] = 1.0
        MA[:, k] = cp.asnumpy(c.apply_M_inv(c.apply_A(e))).ravel()
        e.ravel()[k] = 0.0
    eig = np.linalg.eigvals(MA)
    eig = np.real(eig)          # similar to an SPD matrix — spectrum is real
    assert eig.min() > 0, eig.min()
    spread = eig.max() / eig.min()
    assert spread < 25.0, spread


def test_warm_start_reduces_iterations():
    # Uses the prior preconditioner: with the shifted one, cold solves are
    # already so short that cold-vs-warm counts are not reliably ordered.
    c, b, precision, _, rng = make_conditioner(rtol=1e-5,
                                               preconditioner="prior")
    u0 = cp.asarray(rng.standard_normal((NY, NX)).astype(np.float32)) * SIGMA
    c.correct(u0)
    cold_iters = c.last_iters
    c.correct(u0 + 0.01 * cp.asarray(
        rng.standard_normal((NY, NX)).astype(np.float32)))
    assert c.last_iters < cold_iters


def test_even_alpha_raises():
    prior = make_prior(nu=2)     # alpha = 3, truncated by //2 — inconsistent
    if pytest is not None:
        with pytest.raises(ValueError):
            ConditionedPrior(prior, cp.zeros((NY, NX), dtype=cp.float32),
                             cp.zeros((NY, NX), dtype=cp.float32))
    else:
        try:
            ConditionedPrior(prior, cp.zeros((NY, NX), dtype=cp.float32),
                             cp.zeros((NY, NX), dtype=cp.float32))
        except ValueError:
            pass
        else:
            raise AssertionError("even alpha should raise")


# --------------------------------------------------------------------------- #
# torch autograd

def _skip_no_torch():
    if not _HAVE_TORCH:
        if pytest is not None:
            pytest.skip("torch with CUDA unavailable")
        return True
    return False


def test_torch_forward_matches_cupy():
    if _skip_no_torch():
        return
    c, _, _, _, rng = make_conditioner(warm_start=False)
    u0_np = rng.standard_normal((NY, NX)).astype(np.float32) * SIGMA
    out_cp = c.condition(cp.asarray(u0_np))
    u0_t = torch.tensor(u0_np, device="cuda", requires_grad=True)
    out_t = GGaPPCondition.apply(c, u0_t)
    assert rel_err(cp.asnumpy(out_cp), out_t.detach().cpu().numpy()) < 1e-5


def test_torch_gradient_fd():
    if _skip_no_torch():
        return
    # The map is linear, so a central difference is exact up to solver noise.
    c, _, _, _, rng = make_conditioner(rtol=1e-7, warm_start=False)
    u0_np = rng.standard_normal((NY, NX)).astype(np.float32) * SIGMA
    w = torch.tensor(rng.standard_normal((NY, NX)).astype(np.float32),
                     device="cuda")

    def f(arr):
        t = torch.tensor(arr, device="cuda")
        return float((w * GGaPPCondition.apply(c, t)).sum())

    u0_t = torch.tensor(u0_np, device="cuda", requires_grad=True)
    (w * GGaPPCondition.apply(c, u0_t)).sum().backward()
    g = u0_t.grad.cpu().numpy()

    eps = 0.1
    for _ in range(3):
        d = rng.standard_normal((NY, NX)).astype(np.float32)
        fd = (f(u0_np + eps * d) - f(u0_np - eps * d)) / (2 * eps)
        an = float((g * d).sum())
        assert abs(fd - an) / max(abs(an), 1e-10) < 1e-2


def test_torch_data_override_and_grad():
    if _skip_no_torch():
        return
    c, b, precision, _, rng = make_conditioner(rtol=1e-7, warm_start=False)
    u0 = torch.tensor(rng.standard_normal((NY, NX)).astype(np.float32) * SIGMA,
                      device="cuda", requires_grad=True)
    delta_np = (rng.standard_normal((NY, NX)).astype(np.float32)
                * cp.asnumpy((precision > 0)).astype(np.float32))
    data = torch.tensor(cp.asnumpy(b) + delta_np, device="cuda",
                        requires_grad=True)
    out_pert = GGaPPCondition.apply(c, u0.detach(), data.detach())
    out_base = GGaPPCondition.apply(c, u0.detach())
    shift_ref = c.solve(c.precision * cp.asarray(delta_np))[0]
    assert rel_err(cp.asarray((out_pert - out_base).cpu().numpy()),
                   shift_ref) < 1e-2

    w = torch.tensor(rng.standard_normal((NY, NX)).astype(np.float32),
                     device="cuda")
    (w * GGaPPCondition.apply(c, u0, data)).sum().backward()
    # g_u0 + g_data = w exactly (S + (Q+D)^{-1}D = I).
    total = (u0.grad + data.grad).cpu().numpy()
    assert np.allclose(total, w.cpu().numpy(), atol=1e-3 * SIGMA)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for fn in fns:
            fn()
            print(f"PASS {fn.__name__}")
