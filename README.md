# StabPy

A Python package for working with stable distributions, Normal Inverse Gaussian (NIG) distribution, normality testing, and FARIMA processes with long-range dependence.

## Features

### Stable Distributions (`stabpy.stable`)

Complete toolkit for α-stable distributions:

- **Parameter Estimation:**
  - `stabcull` - Fast quantile-based estimation (McCulloch's method)
  - `stabreg` - Accurate regression-based estimation (Koutrouvelis's method)

- **Distribution Functions:**
  - `stabcdf` - Cumulative distribution function (Nolan's algorithm)
  - `stabrnd` - Random number generation (Chambers-Mallows-Stuck method)

- **Goodness-of-Fit Testing:**
  - `stabtest` - Multiple tests: Kolmogorov-Smirnov, Kuiper, Cramér-von Mises, Watson, Anderson-Darling
  - Monte Carlo p-value estimation

### Normal Distribution Testing (`stabpy.normal`)

Comprehensive normality testing:

- **`normtest`** - Goodness-of-fit tests for normal distribution:
  - Kolmogorov-Smirnov test
  - Kuiper test
  - Cramér-von Mises test
  - Watson test
  - Anderson-Darling test
  - Monte Carlo p-value estimation

### Normal Inverse Gaussian Distribution (`stabpy.nig`)

Full NIG distribution implementation:

- **Distribution Functions:**
  - `nigpdf` - Probability density function
  - `nigcdf` - Cumulative distribution function
  
- **Parameter Estimation:**
  - `nigest` - Maximum likelihood estimation (MLE)
  - `nigloglik` - Log-likelihood function

- **Random Generation:**
  - `nigrnd` - NIG random number generation
  - `invgrnd` - Inverse Gaussian random numbers (mixing distribution)

- **Goodness-of-Fit Testing:**
  - `nigtest` - Multiple tests with Monte Carlo p-values

### FARIMA and Fractional Processes (`stabpy.farima`)

Tools for long-memory processes:

- **Fractional Brownian Motion:**
  - `usg` - Generate fractional Gaussian noise (Davies-Harte algorithm)
  - `gam` - Autocovariance function of fGn

- **FARIMA Processes:**
  - `fftfarima` - Simulate FARIMA(p,d,q) with stable or Gaussian innovations
  - `fftFarimaEst` - Estimate FARIMA parameters (Whittle likelihood)
  - `fracdiff` - Fractional differencing operator coefficients

- **Linear Fractional Stable Noise:**
  - `fftlfsn` - Generate LFSN (generalization of fBm to stable laws)
  - `sstabrnd` - Symmetric stable random numbers

- **Utilities:**
  - `IntegralEst` - Objective function for parameter estimation

## Installation

```bash
pip install stabpy
```

## Quick Start

### Stable Distributions

```python
import numpy as np
from stabpy import stabrnd, stabcull, stabreg, stabtest

# Generate stable random sample
x = stabrnd(alpha=1.5, beta=0.5, m=1000, n=1).flatten()

# Estimate parameters (quantile method - fast)
alpha, sigma, beta, mu = stabcull(x)
print(f"Quantile estimates: α={alpha:.3f}, β={beta:.3f}")

# Estimate parameters (regression method - more accurate)
alpha, sigma, beta, mu = stabreg(x)
print(f"Regression estimates: α={alpha:.3f}, β={beta:.3f}")

# Test goodness of fit
results = stabtest(x, ilp=1000)
print(f"KS p-value: {results[9]:.3f}")
```

### Normality Testing

```python
import numpy as np
from stabpy import normtest

# Test if data comes from normal distribution
x = np.random.randn(200)
results = normtest(x, ilp=1000)

# Extract results
mu, sigma = results[0, 0:2]
test_stats = results[0, 2:7]  # [D, V, W2, U2, A2]
p_values = results[0, 7:12]

print(f"Parameters: μ={mu:.3f}, σ={sigma:.3f}")
print(f"Anderson-Darling p-value: {p_values[4]:.3f}")
```

### Normal Inverse Gaussian

```python
from stabpy import nigrnd, nigest, nigtest

# Generate NIG sample
x = nigrnd(alpha=1.5, beta=0.5, mu=0.0, delta=1.0, m=500, n=1).flatten()

# Estimate parameters via MLE
params = nigest(x)  # [alpha, beta, delta, mu]
print(f"MLE estimates: α={params[0]:.3f}, β={params[1]:.3f}")

# Test fit
results = nigtest(x, ilp=1000)
print(f"KS p-value: {results[9]:.3f}")
```

### FARIMA Processes

```python
from stabpy import fftfarima, fftFarimaEst, usg
import numpy as np

# Generate FARIMA(1, 0.3, 1) with Gaussian innovations
Y = fftfarima(alpha=2.0, d=0.3, n=5, Ph=[0.5], Th=[0.3], M=100, N=500)

# Estimate parameters
params = fftFarimaEst(Y[0, :], p=1, q=1)
print(f"Estimated d={params[0]:.3f}, φ={params[1]:.3f}, θ={params[2]:.3f}")

# Generate fractional Gaussian noise
fgn = usg(H_=0.7, N_=10)  # 2^10 = 1024 samples

# Construct fractional Brownian motion
fbm = np.cumsum(fgn)
```

## Modules

- **`stabpy.stable`**: Stable distribution tools
  - Parameter estimation (quantile and regression methods)
  - CDF computation and random generation
  - Comprehensive goodness-of-fit testing

- **`stabpy.normal`**: Normal distribution goodness-of-fit tests
  - Five different test statistics
  - Monte Carlo p-value estimation
  
- **`stabpy.nig`**: Normal Inverse Gaussian distribution
  - PDF, CDF, and random generation
  - Maximum likelihood parameter estimation
  - Goodness-of-fit testing

- **`stabpy.farima`**: FARIMA processes and fractional Brownian motion
  - fGn and fBm generation
  - FARIMA simulation with stable or Gaussian innovations
  - Parameter estimation via Whittle likelihood

## Documentation

Full documentation available in docstrings:

```python
from stabpy import stabcdf
help(stabcdf)
```

Each function includes:
- Detailed parameter descriptions
- Return value specifications
- Mathematical background and formulas
- Usage examples
- References to scientific literature

## Requirements

- Python ≥ 3.8
- NumPy ≥ 1.20.0
- SciPy ≥ 1.7.0

## Testing

Run tests with pytest:

```bash
pip install pytest
pytest tests/
```

## Examples

See the `examples/` directory for complete examples:
- `example_stable.py` - Stable distribution analysis with visualizations
- `example_nig.py` - NIG distribution fitting and testing
- `example_farima.py` - FARIMA and fBm generation with scaling analysis

## License

MIT License - see LICENSE file for details.

## Authors
J. Kończal (julia.konczal@pwr.edu.pl), M. Balcerek (michal.balcerek@pwr.edu.pl), K. Burnecki (krzysztof.burnecki@pwr.edu.pl)



## Acknowledgments
This package is based on MATLAB implementations developed at the Hugo Steinhaus Center, Wrocław University of Science and Technology, by M. Banys, S. Borak, K. Burnecki, B. Dybiec, M. Lach, A. Misiorek, S. Niedziela, G. Sikora, S. Stoev, S. Szymanski, M. Tempes, R. Werner, and R. Weron.
