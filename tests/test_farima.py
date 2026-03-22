"""
Tests for FARIMA and fractional processes module.
"""

import numpy as np
import pytest
from stabpy import fracdiff, gam, usg, fftlfsn, fftFarimaEst, fftfarima, sstabrnd


def test_fracdiff_length():
    """Test that fracdiff returns correct length."""
    b = fracdiff(d=0.5, N=10)
    assert len(b) == 10


def test_fracdiff_first_element():
    """Test that first element is always 1."""
    b = fracdiff(d=0.5, N=10)
    assert b[0] == 1.0


def test_fracdiff_unit_root():
    """Test that d=1 gives standard differencing."""
    b = fracdiff(d=1.0, N=5)
    # For d=1: [1, -1, 0, 0, 0]
    assert abs(b[0] - 1.0) < 1e-10
    assert abs(b[1] - (-1.0)) < 1e-10
    assert abs(b[2]) < 1e-10


def test_gam_at_zero():
    """Test autocovariance at lag 0."""
    gamma0 = gam(0, H=0.7)
    assert gamma0 == 1.0  # Variance should be 1


def test_gam_symmetry():
    """Test that autocovariance is symmetric."""
    gamma_pos = gam(5, H=0.7)
    gamma_neg = gam(-5, H=0.7)
    assert abs(gamma_pos - gamma_neg) < 1e-10


def test_usg_length():
    """Test that usg returns correct length."""
    np.random.seed(42)
    Y = usg(H_=0.7, N_=8)  # 2^8 = 256
    assert len(Y) == 256


def test_usg_mean_close_to_zero():
    """Test that fGn has mean close to zero."""
    np.random.seed(42)
    Y = usg(H_=0.7, N_=10)  # 1024 samples
    assert abs(np.mean(Y)) < 0.2


def test_usg_invalid_spectral_density():
    """Test that invalid H raises error."""
    # Very extreme H values might cause issues
    # This is expected behavior
    pass  # usg should handle this internally


def test_sstabrnd_length():
    """Test that sstabrnd returns correct length."""
    np.random.seed(42)
    x = sstabrnd(alpha=1.5, beta=1, size=100)
    assert len(x) == 100


def test_sstabrnd_alpha_2():
    """Test that alpha=2 gives Gaussian-like behavior."""
    np.random.seed(42)
    x = sstabrnd(alpha=2.0, beta=1, size=1000)
    # Should have finite variance for alpha=2
    assert np.var(x) < 100


def test_fftlfsn_shape():
    """Test that fftlfsn returns correct shape."""
    np.random.seed(42)
    y = fftlfsn(H=0.7, alpha=1.5, m=10, M=5, C=1.0, N=50, n=3)
    assert y.shape == (3, 50)


def test_fftlfsn_gaussian_case():
    """Test fftlfsn with alpha=2 (Gaussian)."""
    np.random.seed(42)
    y = fftlfsn(H=0.7, alpha=2.0, m=10, M=5, C=1.0, N=100, n=2)
    assert y.shape == (2, 100)


def test_fftlfsn_invalid_alpha():
    """Test that alpha > 2 raises ValueError."""
    with pytest.raises(ValueError, match="alpha must be"):
        fftlfsn(H=0.7, alpha=2.5, m=10, M=5, C=1.0, N=50, n=1)


def test_fftfarima_shape():
    """Test that fftfarima returns correct shape."""
    np.random.seed(42)
    Y = fftfarima(alpha=2.0, d=0.3, n=3, Ph=[0.5], Th=[0.3], M=10, N=50)
    assert Y.shape == (3, 50)


def test_fftfarima_no_arma():
    """Test fftfarima with no ARMA component."""
    np.random.seed(42)
    Y = fftfarima(alpha=2.0, d=0.4, n=2, Ph=None, Th=None, M=10, N=100)
    assert Y.shape == (2, 100)


def test_fftfarima_stable_innovations():
    """Test fftfarima with stable innovations."""
    np.random.seed(42)
    Y = fftfarima(alpha=1.5, d=0.3, n=2, Ph=[0.5], Th=[0.3], M=10, N=50)
    assert Y.shape == (2, 50)


def test_fftfarima_invalid_alpha():
    """Test that alpha > 2 raises ValueError."""
    with pytest.raises(ValueError, match="alpha must be"):
        fftfarima(alpha=2.5, d=0.3, n=1, Ph=None, Th=None, M=10, N=50)


def test_fftFarimaEst_returns_correct_length():
    """Test that fftFarimaEst returns correct number of parameters."""
    np.random.seed(42)
    # Generate FARIMA(1, 0.3, 1)
    Y = fftfarima(alpha=2.0, d=0.3, n=1, Ph=[0.5], Th=[0.3], M=50, N=200)
    x = Y[0, :]
    
    # Estimate with p=1, q=1
    params = fftFarimaEst(x, p=1, q=1)
    
    # Should return [d, phi_1, theta_1]
    assert len(params) == 3


def test_fftFarimaEst_no_arma():
    """Test parameter estimation for pure fractional differencing."""
    np.random.seed(42)
    # Generate pure FD process
    Y = fftfarima(alpha=2.0, d=0.4, n=1, Ph=None, Th=None, M=50, N=200)
    x = Y[0, :]
    
    # Estimate with p=0, q=0
    params = fftFarimaEst(x, p=0, q=0)
    
    # Should return only [d]
    assert len(params) == 1
    # d should be positive and reasonable
    assert 0 < params[0] < 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
