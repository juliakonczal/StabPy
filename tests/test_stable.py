"""
Tests for stable distribution module.
"""

import numpy as np
import pytest
from stabpy import stabcdf, stabcull, stabreg, stabrnd, stabtest


def test_stabrnd_shape():
    """Test that stabrnd returns correct shape."""
    x = stabrnd(alpha=1.5, beta=0.5, m=10, n=5)
    assert x.shape == (10, 5), f"Expected shape (10, 5), got {x.shape}"


def test_stabrnd_symmetric():
    """Test symmetric stable distribution (beta=0)."""
    np.random.seed(42)
    x = stabrnd(alpha=1.5, beta=0.0, m=1000, n=1)
    assert x.shape == (1000, 1)
    # Symmetric should have mean close to 0
    assert abs(np.mean(x)) < 0.2


def test_stabrnd_invalid_alpha():
    """Test that invalid alpha returns NaN."""
    x = stabrnd(alpha=2.5, beta=0.0, m=10, n=1)
    assert np.all(np.isnan(x))


def test_stabrnd_invalid_beta():
    """Test that invalid beta returns NaN."""
    x = stabrnd(alpha=1.5, beta=1.5, m=10, n=1)
    assert np.all(np.isnan(x))


def test_stabcdf_scalar():
    """Test CDF at a single point."""
    cdf = stabcdf(0, alpha=1.5, sigma=1.0, beta=0.0, mu=0.0)
    # For symmetric distribution, CDF(0) should be close to 0.5
    assert 0.4 < cdf < 0.6, f"Expected ~0.5, got {cdf}"


def test_stabcdf_array():
    """Test CDF at multiple points."""
    x = np.array([-1, 0, 1])
    cdf = stabcdf(x, alpha=1.5, sigma=1.0, beta=0.0, mu=0.0)
    assert len(cdf) == 3
    # CDF should be monotonically increasing
    assert cdf[0] < cdf[1] < cdf[2]


def test_stabcdf_cauchy():
    """Test CDF for Cauchy distribution (alpha=1, beta=0)."""
    cdf = stabcdf(0, alpha=1.0, sigma=1.0, beta=0.0, mu=0.0)
    # Cauchy CDF at median should be 0.5
    assert abs(cdf - 0.5) < 0.01


def test_stabcull_basic():
    """Test quantile-based parameter estimation."""
    np.random.seed(42)
    # Generate data with known parameters
    x = stabrnd(alpha=1.5, beta=0.5, m=500, n=1).flatten()
    
    # Estimate parameters
    alpha_est, sigma_est, beta_est, mu_est = stabcull(x)
    
    # Check that estimates are reasonable
    assert 0 < alpha_est <= 2.0
    assert sigma_est > 0
    assert -1 <= beta_est <= 1


def test_stabreg_basic():
    """Test regression-based parameter estimation."""
    np.random.seed(42)
    # Generate data
    x = stabrnd(alpha=1.5, beta=0.3, m=200, n=1).flatten()
    
    # Estimate parameters
    alpha_est, sigma_est, beta_est, mu_est = stabreg(x)
    
    # Check constraints
    assert 0 < alpha_est <= 2.0
    assert sigma_est > 0
    assert -1 <= beta_est <= 1


def test_stabtest_returns_correct_shape():
    """Test that stabtest returns array of correct length."""
    np.random.seed(42)
    x = stabrnd(alpha=1.5, beta=0.5, m=100, n=1).flatten()
    
    results = stabtest(x, ilp=100)
    
    assert len(results) == 14
    # First 4 are parameters
    assert 0 < results[0] <= 2.0  # alpha
    assert results[1] > 0  # sigma
    assert -1 <= results[2] <= 1  # beta
    # Last 5 are p-values (should be between 0 and 1)
    for pval in results[9:14]:
        assert 0 <= pval <= 1


def test_stabreg_converges():
    """Test that stabreg converges for good data."""
    np.random.seed(42)
    # Generate clean data
    x = 2.0 * stabrnd(alpha=1.8, beta=0.0, m=300, n=1).flatten() + 1.0
    
    # Should converge without error
    alpha_est, sigma_est, beta_est, mu_est = stabreg(x)
    
    # Estimates should be reasonable
    assert 1.0 < alpha_est < 2.0
    assert sigma_est > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
