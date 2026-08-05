import warnings

import cupy as cp


def _dot(a, b):
    """float64 inner product of float32 fields (PCG numerics)."""
    return float(cp.sum(a * b, dtype=cp.float64))


class ConditionedPrior:
    """Matérn GP conditioned on pointwise heteroscedastic data (Matheron's rule).

    Wraps a MaternPrior (N(0, C), C = (tau/dx)^2 L^{-alpha}) together with a
    per-pixel diagonal precision field D (zero = unobserved; precisions of
    coincident data add) and the precision-weighted data field b. The
    conditioning map is

        condition(u0) = u0 + (Q + D)^{-1} [D * (b - u0)],   Q = C^{-1},

    which for u0 = mean + prior sample is an exact draw from the GP
    conditioned on the data (Matheron's rule), so a whitened parametrization
    of u0 keeps its standard-normal prior unchanged.

    Solves are matrix-free flexible PCG: the matvec applies Q as alpha cheap
    stencil operations (two `whiten` calls); the preconditioner is either the
    prior C or (default) the diagonally shifted factor inverse
    (dx/tau)^2 (L + d)^-2 — see the `preconditioner` notes in __init__.
    Shifted-preconditioned solves take O(10) iterations regardless of data
    strength; prior-preconditioned counts scale like sigma_prior/sigma_obs.

    Caveats: only odd nu is self-consistent (alpha = nu + 1 must be even —
    enforced); delta != 1 rescales the marginal variance to sigma^2/delta
    (warned); the Neumann boundary conditions inflate the prior variance
    within ~l of the domain edge (not corrected — in typical use the edge
    pixels are data-anchored, which collapses the variance there anyway).
    """

    def __init__(self, prior, data, precision, *,
                 rtol=1e-4, atol=0.0, maxiter=400, warm_start=True,
                 rtol_adjoint=None, preconditioner="shifted"):
        params = prior.mg[prior.top_level].parameters
        alpha = int(params.alpha.value)
        if alpha % 2 != 0:
            raise ValueError(
                f"ConditionedPrior requires even alpha = nu + 1 (odd nu); "
                f"got alpha={alpha}. Even nu silently truncates alpha//2 in "
                f"MaternPrior and is inconsistent with its own tau.")
        delta = float(params.delta.value)
        if delta != 1.0:
            warnings.warn(
                f"delta={delta} != 1: the marginal variance is sigma^2/delta "
                f"and the effective range is l*sqrt(delta).")
        self.prior = prior
        self.data = cp.array(data, dtype=cp.float32)
        self.precision = cp.array(precision, dtype=cp.float32)
        self.rtol = rtol
        self.atol = atol
        self.maxiter = maxiter
        self.warm_start = warm_start
        # Backward-pass solves may use a looser tolerance: a ~1% gradient is
        # ample for stochastic optimization, while rough loss cotangents can
        # stall float32 PCG well before a 1e-4 residual (burning maxiter).
        self.rtol_adjoint = rtol if rtol_adjoint is None else rtol_adjoint
        # Warm-start caches (forward/adjoint solutions from the previous
        # call). Stale caches — e.g. after a data override swap — are merely
        # initial guesses: always correct, just possibly slower.
        self._mu = None
        self._w = None
        self.last_iters = 0
        self.last_converged = True

        # Preconditioner. "prior": M^-1 = C (two multigrid solves of the
        # unshifted factor L) — the preconditioned spectrum carries the full
        # (sigma_prior/sigma_obs)^2 spread, costing O(sigma_prior/sigma_obs)
        # iterations. "shifted" (default): M^-1 = (dx/tau)^2 (L + d)^-2 with
        # the per-cell diagonal d = (tau/dx) sqrt(D) folded into the mass
        # term of a SECOND multigrid hierarchy, so
        # M = Q + D + (dx/tau)^2 (Ld + dL) differs from the target operator
        # only by the cross term — O(10) iterations regardless of data
        # strength or right-hand-side roughness, at identical per-iteration
        # cost. (d is a genuine density coefficient — per-cell precisions
        # are dx^2-scaled precision densities — so it restricts by plain
        # averaging and the hierarchy stays consistent.) Costs one extra
        # hierarchy of memory; only the preconditioner changes, never the
        # solution.
        if preconditioner not in ("shifted", "prior"):
            raise ValueError(f"preconditioner must be 'shifted' or 'prior', "
                             f"got {preconditioner!r}")
        self.preconditioner = preconditioner
        self._shifted = None
        if preconditioner == "shifted" and bool(self.precision.any()):
            self._shifted = self._build_shifted_prior()

    def _build_shifted_prior(self):
        """A MaternPrior whose operator is (L + d), d = (tau/dx)sqrt(D),
        sharing the plain prior's hyperparameters and geometry. The plain
        prior's own hierarchy must stay unshifted (it defines C and the
        whitened map) — the shift enters as separate coefficient state.

        When the wrapped prior is a PriorCollection member (and is rooted at
        the finest level), the shifted operator joins the same collection: it
        reuses the shared hierarchy/solver and owns only its (ny, nx) shift
        arrays, instead of a second full multigrid stack. Otherwise a private
        stack is built exactly as before."""
        base = self.prior.mg[self.prior.top_level]
        d = (float(base.parameters.tau.value) / float(base.dx)) \
            * cp.sqrt(self.precision)

        collection = getattr(self.prior, "_collection", None)
        if collection is not None and self.prior.top_level == 0:
            name = f"__shifted:{self.prior.name}:{id(self):x}"
            return collection.add(
                name,
                float(base.parameters.sigma.value),
                float(base.parameters.l.value),
                int(base.parameters.nu.value),
                delta=float(base.parameters.delta.value),
                shift=d)

        from .model import MaternPrior
        from .multigrid import Multigrid
        smg = Multigrid(len(self.prior.mg.levels) - self.prior.top_level,
                        ny=int(base.ny), nx=int(base.nx), dx=float(base.dx))
        for name in ("sigma", "l", "nu", "delta"):
            getattr(smg.parameters, name).set(
                float(getattr(base.parameters, name).value))
        smg.coefficients.shift.set(d)
        shifted = MaternPrior(mg=smg)
        shifted.forward_solver.fas_options.report_norms.set(False)
        return shifted

    # --- linear operators, composed from the prior's own entry points so the
    # --- (tau/dx) scalings live in exactly one place (model.py)
    def apply_Q(self, v):
        return self.prior.whiten(self.prior.whiten(v))

    def apply_C(self, v):
        return self.prior.forward(self.prior.forward(v, zero_init=True),
                                  zero_init=True)

    def apply_M_inv(self, v):
        """The PCG preconditioner: C, or the shifted-factor inverse."""
        if self._shifted is None:
            return self.apply_C(v)
        return self._shifted.forward(self._shifted.forward(v, zero_init=True),
                                     zero_init=True)

    def apply_A(self, v):
        return self.apply_Q(v) + self.precision * v

    def solve(self, rhs, x0=None, rtol=None):
        """Flexible PCG (Polak-Ribiere beta) for (Q + D) x = rhs.

        The preconditioner is an inexact, iteration-dependent multigrid solve,
        so plain PCG's beta is not safe; flexible CG is, at the cost of one
        stored vector. Dot products accumulate in float64; the true residual
        is recomputed every 50 iterations. Convergence is measured against
        ||rhs|| (not the initial residual) so warm starts cannot fake it.
        Returns (x, n_iter, converged).
        """
        rtol = self.rtol if rtol is None else rtol
        rhs_norm = _dot(rhs, rhs) ** 0.5
        tol = max(rtol * rhs_norm, self.atol)
        if rhs_norm == 0.0:
            return cp.zeros_like(rhs), 0, True

        if x0 is not None:
            x = x0.copy()
            r = rhs - self.apply_A(x)
        else:
            x = cp.zeros_like(rhs)
            r = rhs.copy()

        z = self.apply_M_inv(r)
        p = z.copy()
        rz = _dot(r, z)
        converged = _dot(r, r) ** 0.5 <= tol
        it = 0
        while not converged and it < self.maxiter:
            Ap = self.apply_A(p)
            pAp = _dot(p, Ap)
            if pAp <= 0.0:
                warnings.warn(f"ConditionedPrior.solve: non-positive curvature "
                              f"(<p,Ap>={pAp:.3e}) at iteration {it}; stopping.")
                break
            step = rz / pAp
            x += cp.float32(step) * p
            r_prev = r.copy()
            r = r - cp.float32(step) * Ap
            it += 1
            if it % 50 == 0:
                r = rhs - self.apply_A(x)
            if _dot(r, r) ** 0.5 <= tol:
                converged = True
                break
            z = self.apply_M_inv(r)
            beta = _dot(z, r - r_prev) / rz
            rz = _dot(r, z)
            p = z + cp.float32(beta) * p

        if not converged:
            achieved = _dot(r, r) ** 0.5 / rhs_norm
            warnings.warn(
                f"ConditionedPrior.solve: not converged after {it} iterations "
                f"(relative residual {achieved:.2e}, target {rtol:.1e}); "
                f"proceeding with the current iterate.")
        self.last_iters = it
        self.last_converged = converged
        return x, it, converged

    def correct(self, u0, data=None):
        """mu = (Q + D)^{-1} [D * (b - u0)], warm-started from the previous mu."""
        b = self.data if data is None else data
        rhs = self.precision * (b - u0)
        mu, _, _ = self.solve(rhs, x0=self._mu if self.warm_start else None)
        if self.warm_start:
            self._mu = mu
        return mu

    def condition(self, u0, data=None):
        return u0 + self.correct(u0, data=data)

    def adjoint_solve(self, gbar):
        """w = (Q + D)^{-1} gbar (the backward-pass solve), warm-started,
        at the (typically looser) adjoint tolerance."""
        w, _, _ = self.solve(gbar, x0=self._w if self.warm_start else None,
                             rtol=self.rtol_adjoint)
        if self.warm_start:
            self._w = w
        return w

    def latent_from_field(self, field, mean):
        """The fluctuation x such that condition(mean + x) == field, exactly:
        since condition(u0) = S u0 + mu_b with S = (Q+D)^{-1} Q and
        S^{-1} = I + C D,   u0 = (I + C D)(field - mu_b),  x = u0 - mean.
        Used for exact checkpoint conversion between parametrizations."""
        mu_b, _, _ = self.solve(self.precision * self.data)
        v = field - mu_b
        return v + self.apply_C(self.precision * v) - mean
