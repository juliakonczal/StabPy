"""
Tests for normal distribution testing module.
"""

import numpy as np
import pytest
from stabpy import normtest


def test_normtest_gaussian_data():
    """Test normtest on data from normal distribution."""
    np.random.seed(42)
    x = np.random.randn(100)
    
    results = normtest(x, ilp=100)
    
    # Check shape
    assert results.shape == (1, 12)
    
    # Check parameter estimates
    mu_est = results[0, 0]
    sigma_est = results[0, 1]
    assert abs(mu_est) < 0.5  # Should be close to 0
    assert 0.5 < sigma_est < 1.5  # Should be close to 1
    
    # Check p-values (should be high for normal data)
    # At least some p-values should be > 0.05
    pvalues = results[0, 7:12]
    assert np.any(pvalues > 0.05)


def test_normtest_returns_correct_shape():
    """Test that normtest returns correct output shape."""
    np.random.seed(42)
    x = np.random.randn(50)
    
    results = normtest(x, ilp=50)
    
    assert results.shape == (1, 12)


def test_normtest_statistics_positive():
    """Test that test statistics are positive."""
    np.random.seed(42)
    x = np.random.randn(100)
    
    results = normtest(x, ilp=100)
    
    # Test statistics should be positive
    for i in range(2, 7):
        assert results[0, i] >= 0


def test_normtest_pvalues_in_range():
    """Test that p-values are between 0 and 1."""
    np.random.seed(42)
    x = np.random.randn(100)
    
    results = normtest(x, ilp=100)
    
    # P-values should be in [0, 1]
    for i in range(7, 12):
        assert 0 <= results[0, i] <= 1


def test_normtest_heavy_tailed_data():
    """Test normtest on heavy-tailed data (should reject)."""
    np.random.seed(42)
    # Student's t with 3 df (heavy tails)
    from scipy.stats import t
    x = t.rvs(df=3, size=200)
    
    results = normtest(x, ilp=200)
    
    # Should have some low p-values (reject normality)
    pvalues = results[0, 7:12]
    # At least one test should reject (p < 0.05)
    assert np.any(pvalues < 0.05)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
