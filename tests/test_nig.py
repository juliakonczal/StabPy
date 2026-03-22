"""
Tests for Normal Inverse Gaussian distribution module.
"""

import numpy as np
import pytest
from stabpy import nigpdf, nigcdf, nigest, nigrnd, nigtest, invgrnd


def test_nigrnd_shape():
    """Test that nigrnd returns correct shape."""
    x = nigrnd(alpha=1.5, beta=0.5, mu=0.0, delta=1.0, m=10, n=5)
    assert x.shape == (10, 5)


def test_nigrnd_symmetric():
    """Test symmetric NIG distribution (beta=0)."""
    np.random.seed(42)
    x = nigrnd(alpha=1.5, beta=0.0, mu=0.0, delta=1.0, m=1000, n=1)
    # Symmetric should have mean close to mu
    assert abs(np.mean(x)) < 0.2


def test_nigrnd_invalid_alpha():
    """Test that invalid alpha raises ValueError."""
    with pytest.raises(ValueError, match="ALPHA must be positive"):
        nigrnd(alpha=-1.0, beta=0.0, mu=0.0, delta=1.0, m=10, n=1)


def test_nigrnd_invalid_delta():
    """Test that invalid delta raises ValueError."""
    with pytest.raises(ValueError, match="DELTA must be positive"):
        nigrnd(alpha=1.5, beta=0.0, mu=0.0, delta=-1.0, m=10, n=1)


def test_nigrnd_invalid_beta():
    """Test that beta >= alpha raises ValueError."""
    with pytest.raises(ValueError, match="BETA must be smaller than ALPHA"):
        nigrnd(alpha=1.0, beta=1.5, mu=0.0, delta=1.0, m=10, n=1)


def test_nigpdf_symmetric():
    """Test PDF at center for symmetric distribution."""
    pdf = nigpdf(0, alpha=1.5, beta=0.0, delta=1.0, mu=0.0)
    assert pdf > 0
    assert not np.isnan(pdf)


def test_nigpdf_monotonic_for_symmetric():
    """Test that PDF is symmetric around mu."""
    x = np.array([-1, 0, 1])
    pdf = nigpdf(x, alpha=1.5, beta=0.0, delta=1.0, mu=0.0)
    # For symmetric distribution, pdf(-x) should equal pdf(x)
    assert abs(pdf[0] - pdf[2]) < 0.01


def test_nigcdf_monotonic():
    """Test that CDF is monotonically increasing."""
    x = np.array([-2, -1, 0, 1, 2])
    cdf = nigcdf(x, alpha=1.5, beta=0.5, delta=1.0, mu=0.0)
    # CDF should be increasing
    for i in range(len(cdf) - 1):
        assert cdf[i] <= cdf[i+1]


def test_nigcdf_bounds():
    """Test that CDF is between 0 and 1."""
    x = np.array([-5, 0, 5])
    cdf = nigcdf(x, alpha=1.5, beta=0.5, delta=1.0, mu=0.0)
    assert np.all(cdf >= 0)
    assert np.all(cdf <= 1)


def test_nigest_returns_four_params():
    """Test that nigest returns 4 parameters."""
    np.random.seed(42)
    x = nigrnd(alpha=1.5, beta=0.5, mu=0.0, delta=1.0, m=100, n=1).flatten()
    
    params = nigest(x)
    
    assert len(params) == 4
    alpha, beta, delta, mu = params
    assert alpha > abs(beta)
    assert delta > 0


def test_nigest_with_initial_values():
    """Test nigest with custom initial values."""
    np.random.seed(42)
    x = nigrnd(alpha=2.0, beta=0.3, mu=1.0, delta=1.5, m=100, n=1).flatten()
    
    params = nigest(x, x0=[1.5, 0.0, 1.0])
    
    assert len(params) == 4


def test_nigtest_returns_correct_shape():
    """Test that nigtest returns array of correct length."""
    np.random.seed(42)
    x = nigrnd(alpha=1.5, beta=0.5, mu=0.0, delta=1.0, m=100, n=1).flatten()
    
    results = nigtest(x, ilp=50)
    
    assert len(results) == 14
    # First 4 are parameters
    assert results[0] > abs(results[1])  # alpha > |beta|
    assert results[2] > 0  # delta > 0
    # Last 5 are p-values
    for pval in results[9:14]:
        assert 0 <= pval <= 1


def test_invgrnd_shape():
    """Test that invgrnd returns correct shape."""
    x = invgrnd(delta=1.0, gamma=2.0, M=10, N=5)
    assert x.shape == (10, 5)


def test_invgrnd_positive():
    """Test that inverse Gaussian values are positive."""
    np.random.seed(42)
    x = invgrnd(delta=1.0, gamma=2.0, M=100, N=1)
    assert np.all(x > 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
