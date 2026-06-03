# ggapp

**GPU-accelerated Gaussian Process Prior**

`ggapp` provides a fast, GPU-accelerated implementation of Gaussian
process (GP) priors with a Matérn covariance, built for large
spatial problems where forming a dense covariance matrix is
infeasible. Rather than working with the covariance directly, it uses
the **SPDE representation** of the Matérn field: a sample from a
Matérn GP is obtained by repeatedly solving an elliptic
(Laplacian-like) PDE driven by white noise. These solves are carried
out on a structured grid with a geometric **multigrid** /
Full Approximation Scheme (FAS) solver, with the heavy numerical
kernels implemented as custom CUDA kernels and executed on the GPU
through [CuPy](https://cupy.dev/).

The result is a prior that supports the operations needed for
Bayesian inference at scale:

- **`sample()`** — draw a realization from the Matérn prior.
- **`forward(z)`** — map a white-noise/whitened field to a sample
  from the prior (the square-root covariance applied to `z`).
- **`whiten(x)`** — the inverse map, transforming a field into the
  whitened space in which the prior is standard-normal.
- **`log_probability(x)`** — evaluate the (unnormalized) prior
  log-density.

These maps are exposed as differentiable PyTorch operations via the
optional `ggapp.torch` module, so a `ggapp` prior can be dropped into
a gradient-based inference or optimization pipeline.

The library is built on top of [`glide`](https://pypi.org/project/glide/),
which supplies the grid and field abstractions used throughout.

## Requirements

`ggapp` runs on the GPU and requires:

- **Python** ≥ 3.10
- An **NVIDIA GPU** with a **CUDA 12.x** toolkit/driver
  (needed by `cupy-cuda12x`)

## Dependencies

### Core (always required)

| Package          | Constraint        | Used for                                              |
| ---------------- | ----------------- | ----------------------------------------------------- |
| `cupy-cuda12x`   | `>=12.0.0`        | GPU arrays, custom CUDA kernels (`cupy`, `cupyx`)     |
| `glide`          | —                 | Grid / field abstractions (`glide.field`)             |

### Optional extras

| Extra       | Package(s)                       | Used for                                                     |
| ----------- | -------------------------------- | ----------------------------------------------------------- |
| `torch`     | `torch>=2.0`                     | Differentiable PyTorch autograd integration (`ggapp.torch`) |
| `examples`  | `matplotlib>=3.5`, `scipy>=1.7`  | Running the scripts under `examples/`                       |

These are declared in
[`pyproject.toml`](pyproject.toml) and mirrored in
[`requirements.txt`](requirements.txt) (core) and
[`requirements-optional.txt`](requirements-optional.txt) (extras).

## Installation

Install the core package:

```bash
pip install .
```

Install with optional extras:

```bash
pip install ".[torch]"        # PyTorch integration
pip install ".[examples]"     # to run examples/
pip install ".[torch,examples]"
```

Or with the requirements files:

```bash
pip install -r requirements.txt                               # core only
pip install -r requirements.txt -r requirements-optional.txt  # everything
```

## Quick start

```python
from ggapp.model import MaternPrior

# A 2D Matern prior on a 256 x 256 grid with 5 multigrid levels.
prior = MaternPrior(n_levels=5, ny=256, nx=256, dx=1.0)

x = prior.sample()            # draw a realization from the prior
z = prior.whiten(x)           # map to the whitened (standard-normal) space
x_back = prior.forward(z)     # map back to a prior sample
logp = prior.log_probability(x)
```

See [`examples/toy/sample_gp.py`](examples/toy/sample_gp.py) for a
runnable example (requires the `examples` extra).

## License

See [LICENSE](LICENSE).
