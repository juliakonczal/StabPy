"""
FARIMA and Fractional Processes
================================

This module provides tools for working with Fractionally Integrated ARMA (FARIMA)
processes and fractional Brownian motion (fBm). These processes exhibit long-range
dependence and are useful for modeling phenomena with long memory, such as financial
time series, network traffic, and hydrology.

Key features:
- Fractional differencing operators
- Fractional Gaussian noise and fractional Brownian motion generation
- Linear fractional stable noise simulation
- FARIMA process simulation and parameter estimation

References
----------
.. [1] J. Beran (1994) "Statistics for Long-Memory Processes", Chapman & Hall.
.. [2] P. Doukhan, G. Oppenheim, M.S. Taqqu (2003) "Theory and Applications of 
       Long-Range Dependence", Birkhäuser.
.. [3] C.W.J. Granger, R. Joyeux (1980) "An introduction to long-memory time 
       series models and fractional differencing", Journal of Time Series 
       Analysis, 1(1), 15-29.
.. [4] A.I. McLeod, K.W. Hipel (1978) "Preservation of the rescaled adjusted 
       range", Water Resources Research, 14(3), 491-518.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.fft import fft, ifft
from scipy.signal import lfilter


def fracdiff(d, N):
    """
    Compute coefficients of fractional differencing operator.
    
    Calculates the binomial coefficients for the fractional differencing
    operator (1-L)^d, where L is the lag operator. These coefficients are
    used in FARIMA model simulation.
    
    Parameters
    ----------
    d : float
        Fractional differencing parameter. Common values:
        - d = 0: no differencing (stationary ARMA)
        - 0 < d < 0.5: long memory, stationary
        - d = 0.5: boundary case
        - 0.5 < d < 1: long memory, non-stationary but mean-reverting
        - d = 1: unit root (standard differencing)
    N : int
        Number of coefficients to compute. Larger N gives more accurate
        approximation but uses more memory.
    
    Returns
    -------
    b : ndarray
        Array of length N containing the fractional differencing coefficients.
        b[0] = 1, and subsequent values follow the binomial expansion.
    
    Notes
    -----
    The fractional differencing operator is defined by its expansion:
    
    .. math::
        (1-L)^d = \\sum_{k=0}^{\\infty} \\binom{d}{k} (-L)^k
    
    where the binomial coefficients are computed recursively:
    
    .. math::
        b_0 = 1, \\quad b_k = b_{k-1} \\cdot \\frac{k-1+d}{k}
    
    **Long memory interpretation:**
    - For 0 < d < 0.5: positive autocorrelations decay hyperbolically
    - For -0.5 < d < 0: negative autocorrelations (anti-persistence)
    - Larger |d| means stronger long-range dependence
    
    **Computational note:**
    The recursive formula is numerically stable and efficient for moderate N.
    For very large N (>10000), consider using logarithms to avoid overflow.
    
    Examples
    --------
    >>> import numpy as np
    >>> from stabpy.farima import fracdiff
    >>> # Standard differencing (d=1)
    >>> b1 = fracdiff(1.0, 5)
    >>> print(b1)  # [1, -1, 0, 0, 0]
    >>> 
    >>> # Fractional differencing (long memory)
    >>> b_frac = fracdiff(0.4, 10)
    >>> print(b_frac[:5])  # Decays slowly
    >>> 
    >>> # Plot decay of coefficients
    >>> import matplotlib.pyplot as plt
    >>> b = fracdiff(0.3, 100)
    >>> plt.plot(np.abs(b))
    >>> plt.yscale('log')
    >>> plt.xlabel('Lag')
    >>> plt.ylabel('|Coefficient|')
    >>> plt.title('Fractional differencing coefficients (d=0.3)')
    
    See Also
    --------
    fftfarima : Uses these coefficients for FARIMA simulation
    
    References
    ----------
    .. [1] C.W.J. Granger, R. Joyeux (1980) "An introduction to long-memory 
           time series models and fractional differencing"
    """
    # Generate indices for recursive computation
    j_1 = np.arange(1, N)
    
    # Compute numerators: (j-1) + d for j = 1, 2, ..., N-1
    j_d = (j_1 - 1 + d)
    
    # Recursively compute binomial coefficients
    # b[0] = 1, b[k] = b[k-1] * (k-1+d) / k
    b = np.concatenate(([1], np.cumprod(j_d / j_1)))
    
    return b


def gam(t, H):
    """
    Autocovariance function of fractional Gaussian noise.
    
    Computes the theoretical autocovariance at lag t for fractional Gaussian
    noise (fGn) with Hurst parameter H. This is used in the circulant embedding
    method for generating fGn.
    
    Parameters
    ----------
    t : array_like
        Lag values at which to evaluate the autocovariance.
        Can be scalar or array.
    H : float
        Hurst parameter. Must be in (0, 1).
        - H = 0.5: standard Brownian motion (no correlation)
        - 0.5 < H < 1: positive correlation (persistent)
        - 0 < H < 0.5: negative correlation (anti-persistent)
    
    Returns
    -------
    gamma : ndarray
        Autocovariance values at lags t.
    
    Notes
    -----
    The autocovariance function of fGn is:
    
    .. math::
        \\gamma(t) = \\frac{1}{2}(|t+1|^{2H} - 2|t|^{2H} + |t-1|^{2H})
    
    **Properties:**
    - gamma(0) = 1 (variance)
    - For H > 0.5: gamma(t) > 0 for all t (long-range positive dependence)
    - For H < 0.5: gamma(t) < 0 for t > 0 (anti-persistence)
    - For H = 0.5: gamma(t) = 0 for t ≠ 0 (white noise)
    
    **Usage in simulation:**
    This function is used in the circulant embedding method (Davies-Harte algorithm)
    to construct the covariance matrix for exact fGn generation.
    
    Examples
    --------
    >>> import numpy as np
    >>> from stabpy.farima import gam
    >>> # Autocovariance at different lags
    >>> lags = np.arange(0, 10)
    >>> gamma_persistent = gam(lags, H=0.7)  # Persistent
    >>> gamma_antipersistent = gam(lags, H=0.3)  # Anti-persistent
    >>> 
    >>> # Plot autocorrelation function
    >>> import matplotlib.pyplot as plt
    >>> lags = np.arange(0, 50)
    >>> for H in [0.3, 0.5, 0.7, 0.9]:
    ...     acf = gam(lags, H) / gam(0, H)  # Normalize
    ...     plt.plot(lags, acf, label=f'H={H}')
    >>> plt.xlabel('Lag')
    >>> plt.ylabel('Autocorrelation')
    >>> plt.legend()
    >>> plt.title('fGn Autocorrelation Functions')
    
    See Also
    --------
    usg : Uses this function to generate fractional Gaussian noise
    
    References
    ----------
    .. [1] B.B. Mandelbrot, J.W. Van Ness (1968) "Fractional Brownian motions, 
           fractional noises and applications", SIAM Review, 10(4), 422-437.
    """
    return 0.5 * (np.abs(t + 1)**(2*H) - 2*np.abs(t)**(2*H) + np.abs(t - 1)**(2*H))


def usg(H_, N_):
    """
    Generate fractional Gaussian noise using FFT method.
    
    Implements the Davies-Harte circulant embedding algorithm to generate
    exact samples of fractional Gaussian noise (fGn). This is an efficient
    FFT-based method that produces exact realizations (not approximations).
    
    Parameters
    ----------
    H_ : float
        Hurst parameter. Must be in (0, 1).
        - H = 0.5: white noise (independent increments)
        - 0.5 < H < 1: long-range positive dependence
        - 0 < H < 0.5: anti-persistence (negative dependence)
    N_ : int
        Log2 of sample size. Output will have length 2^N_.
        For example: N_=10 gives 1024 samples.
    
    Returns
    -------
    Y : ndarray
        Array of length 2^N_ containing fGn sample.
        Has mean ≈ 0 and variance ≈ 1.
    
    Raises
    ------
    ValueError
        If the spectral density has negative values, indicating the
        algorithm failed (very rare, typically only for extreme H values
        combined with insufficient N_).
    
    Notes
    -----
    **Fractional Gaussian Noise (fGn):**
    
    fGn is a stationary increment process of fractional Brownian motion (fBm).
    If B_H(t) is fBm, then X_i = B_H(i) - B_H(i-1) is fGn.
    
    **Algorithm (Davies-Harte):**
    
    1. Construct circulant covariance matrix using autocovariance function
    2. Compute eigenvalues via FFT (these are the spectral densities)
    3. Generate complex Gaussian random variables
    4. Scale by sqrt(eigenvalues) and inverse FFT
    5. Extract real part
    
    **Advantages:**
    - Exact sampling (not approximation)
    - O(N log N) complexity via FFT
    - Works for any H in (0,1)
    
    **Applications:**
    - Modeling long-range dependent time series
    - Simulating self-similar network traffic
    - Generating rough surfaces/textures
    - Financial modeling with persistent volatility
    
    Examples
    --------
    >>> import numpy as np
    >>> from stabpy.farima import usg
    >>> # Generate fGn with positive dependence
    >>> np.random.seed(42)
    >>> fgn_persistent = usg(H_=0.7, N_=10)  # 1024 samples
    >>> print(f"Length: {len(fgn_persistent)}")
    >>> print(f"Mean: {np.mean(fgn_persistent):.4f}")
    >>> print(f"Std: {np.std(fgn_persistent):.4f}")
    >>> 
    >>> # Generate fGn with anti-persistence
    >>> fgn_anti = usg(H_=0.3, N_=10)
    >>> 
    >>> # Construct fractional Brownian motion by cumulative sum
    >>> fbm = np.cumsum(fgn_persistent)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(fbm)
    >>> plt.title('Fractional Brownian Motion (H=0.7)')
    >>> plt.xlabel('Time')
    >>> plt.ylabel('Value')
    
    See Also
    --------
    gam : Autocovariance function used in this algorithm
    fftlfsn : Generates linear fractional stable noise (generalization)
    
    References
    ----------
    .. [1] R.B. Davies, D.S. Harte (1987) "Tests for Hurst effect", Biometrika, 
           74(1), 95-101.
    .. [2] A.T.A. Wood, G. Chan (1994) "Simulation of stationary Gaussian 
           processes in [0,1]^d", Journal of Computational and Graphical 
           Statistics, 3(4), 409-432.
    """
    H = H_
    N = 2**N_  # Actual sample size
    M = 2 * N  # Circulant embedding size
    
    # Construct lag indices for circulant matrix
    t1 = np.arange(0, M//2 + 1)
    t2 = np.arange(M//2 - 1, 0, -1)
    
    # Initialize complex array for frequency domain
    V = np.zeros(M, dtype=complex)
    
    # Compute autocovariance at all required lags
    gamma = gam(np.concatenate((t1, t2), axis=None), H)
    
    # Compute eigenvalues via FFT (spectral density)
    Sc = np.fft.fft(gamma, M)
    S = np.real(Sc[:M//2 + 1])
    
    # Check for non-negative spectral density
    # (required for valid covariance structure)
    if np.any(S < 0):
        raise ValueError('Error in algorithm: negative spectral density detected. '
                        'Try increasing N_ or using H closer to 0.5.')
    
    # Generate independent standard normal random variables
    W = np.random.randn(M)
    
    # Construct complex Gaussian variables in frequency domain
    # Scaled by sqrt of eigenvalues (spectral density)
    V[0] = np.sqrt(S[0]) * W[0]
    V[1:M//2] = np.sqrt(S[1:M//2] / 2) * (W[1:M-1:2] + 1j * W[2:M:2])
    V[M//2] = np.sqrt(S[M//2]) * W[M-1]
    # Enforce conjugate symmetry for real output
    V[M//2+1:] = np.conj(V[M//2-1:0:-1])
    
    # Inverse FFT to get time domain realization
    Yc = 1/np.sqrt(M) * np.fft.fft(V, M)
    
    # Extract real part (first N values)
    Y = np.real(Yc[:N])
    
    return Y


def IntegralEst(beta, p, q, y, P):
    """
    Objective function for FARIMA parameter estimation.
    
    Computes the integrated periodogram-based objective function for estimating
    FARIMA(p,d,q) parameters. This function is minimized to obtain Whittle
    estimates of the model parameters.
    
    Parameters
    ----------
    beta : array_like
        Parameter vector [d, phi_1, ..., phi_p, theta_1, ..., theta_q].
        - beta[0]: fractional differencing parameter d
        - beta[1:p+1]: AR coefficients phi
        - beta[p+1:p+q+1]: MA coefficients theta
    p : int
        Order of AR component.
    q : int
        Order of MA component.
    y : ndarray
        Frequency grid points at which to evaluate spectral density.
        Typically: y = 2*pi*k/N for k = 1, ..., N/2.
    P : ndarray
        Periodogram ordinates at frequencies y.
    
    Returns
    -------
    Y : float
        Integrated objective function value.
        Whittle estimate minimizes this quantity.
    
    Notes
    -----
    **Whittle likelihood approximation:**
    
    The Whittle likelihood for FARIMA models approximates the Gaussian
    likelihood using the periodogram and spectral density:
    
    .. math::
        L(\\theta) \\approx \\sum_{k} \\left[\\log g_k(\\theta) + \\frac{I_k}{g_k(\\theta)}\\right]
    
    where I_k is the periodogram and g_k is the theoretical spectral density.
    
    **FARIMA spectral density:**
    
    For FARIMA(p,d,q):
    
    .. math::
        g(\\omega) = \\frac{|1 + \\sum \\theta_j e^{-ij\\omega}|^2}
                          {|1 + \\sum \\phi_j e^{-ij\\omega}|^2} 
                     \\cdot |2\\sin(\\omega/2)|^{-2d}
    
    **Numerical integration:**
    Uses trapezoidal rule to approximate the integral over [0, π].
    
    Examples
    --------
    >>> import numpy as np
    >>> from stabpy.farima import IntegralEst
    >>> # Simulate data and compute periodogram
    >>> x = np.random.randn(100)
    >>> X = np.fft.fft(x)
    >>> P = np.abs(X[1:51])**2
    >>> y = (2*np.pi/100) * np.arange(1, 51)
    >>> # Evaluate objective at some parameters
    >>> beta = [0.3, 0.5, -0.2]  # d, phi_1, theta_1
    >>> obj = IntegralEst(beta, p=1, q=1, y=y, P=P)
    
    See Also
    --------
    fftFarimaEst : Minimizes this function to estimate parameters
    
    References
    ----------
    .. [1] P. Whittle (1953) "Estimation and information in stationary time series"
    .. [2] M.J. Reisen (1994) "Estimation of the fractional difference parameter 
           in the ARIMA(p,d,q) model using the smoothed periodogram"
    """
    # Extract parameters
    d = beta[0]
    Ph = beta[1:p+1]      # AR coefficients
    Th = beta[p+1:p+q+1]  # MA coefficients
    
    # Compute MA polynomial contribution to spectral density
    # |1 + theta_1*e^(-iw) + ... + theta_q*e^(-iqw)|^2
    N = np.fft.fft(np.concatenate(([0], -Th, np.zeros(2*len(y) - q - 1))))
    N = N[1:]  # Remove DC component
    N = np.abs(1 + N[:len(y)])**2
    
    # Compute AR polynomial contribution to spectral density
    # |1 + phi_1*e^(-iw) + ... + phi_p*e^(-ipw)|^2
    D = np.fft.fft(np.concatenate(([0], -Ph, np.zeros(2*len(y) - p - 1))))
    D = D[1:]  # Remove DC component
    D = np.abs(1 + D[:len(y)])**2
    
    # Compute theoretical spectral density
    # g(w) = (MA part / AR part) * |2sin(w/2)|^(-2d)
    g = N / D * ((2 - 2 * np.cos(y))**(-d))
    
    # Compute ratio of periodogram to spectral density
    f = P / g
    
    # Integrate using trapezoidal rule
    a = y[0]
    b = y[-1]
    n = len(y) - 1
    Y = (b - a) / (2 * n) * (f[0] + f[-1] + 2 * np.sum(f[1:-1]))
    
    return Y


def fftlfsn(H, alpha, m, M, C, N, n):
    """
    Generate linear fractional stable noise using FFT method.
    
    Simulates linear fractional stable noise (LFSN), which generalizes
    fractional Gaussian noise to stable distributions. LFSN exhibits both
    long-range dependence (controlled by H) and heavy tails (controlled by alpha).
    
    Parameters
    ----------
    H : float
        Self-similarity parameter (related to Hurst parameter).
        Typically in (0, 1). Controls long-range dependence.
        - H = 1/alpha: independent increments
        - H > 1/alpha: long-range dependence
    alpha : float
        Stability index. Must be in (0, 2].
        - alpha = 2: Gaussian case (reduces to fGn)
        - alpha < 2: heavy-tailed stable distribution
        Smaller alpha means heavier tails.
    m : int
        Time discretization parameter. Higher values give finer discretization.
    M : int
        Number of past values to include in moving average representation.
        Larger M gives more accurate long-range dependence.
    C : float
        Scale parameter. Controls the overall scale of the process.
    N : int
        Number of time points to generate per realization.
    n : int
        Number of independent realizations to generate.
    
    Returns
    -------
    y : ndarray
        Array of shape (n, N) containing n independent LFSN realizations,
        each of length N.
    
    Raises
    ------
    ValueError
        If alpha > 2.
    
    Notes
    -----
    **Linear Fractional Stable Noise (LFSN):**
    
    LFSN is defined as a moving average of stable white noise:
    
    .. math::
        X_t = \\int_{-\\infty}^{\\infty} 
              \\left[(t-s)_+^{H-1/\\alpha} - (-s)_+^{H-1/\\alpha}\\right] dM_s
    
    where M is a symmetric alpha-stable random measure.
    
    **Properties:**
    - Self-similar with index H
    - Stationary increments
    - Heavy tails (when alpha < 2)
    - Long-range dependence (when H > 1/alpha)
    
    **Special cases:**
    - alpha = 2: fractional Brownian motion
    - H = 1/alpha: Lévy motion (independent increments)
    
    **Algorithm:**
    Uses FFT-based convolution to efficiently compute the moving average
    representation with stable innovations.
    
    Examples
    --------
    >>> import numpy as np
    >>> from stabpy.farima import fftlfsn
    >>> # Generate LFSN with long memory and heavy tails
    >>> np.random.seed(42)
    >>> lfsn = fftlfsn(H=0.7, alpha=1.5, m=10, M=5, C=1.0, N=100, n=5)
    >>> print(f"Shape: {lfsn.shape}")  # (5, 100)
    >>> 
    >>> # Compare Gaussian (alpha=2) vs heavy-tailed (alpha=1.5)
    >>> lfsn_gaussian = fftlfsn(H=0.7, alpha=2.0, m=10, M=5, C=1.0, N=1000, n=1)
    >>> lfsn_heavytail = fftlfsn(H=0.7, alpha=1.5, m=10, M=5, C=1.0, N=1000, n=1)
    >>> 
    >>> # Visualize
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(np.cumsum(lfsn_gaussian[0]), label='Gaussian (α=2)')
    >>> plt.plot(np.cumsum(lfsn_heavytail[0]), label='Heavy-tailed (α=1.5)')
    >>> plt.legend()
    >>> plt.title('Cumulative LFSN')
    
    See Also
    --------
    sstabrnd : Generates stable random variables used in this function
    usg : Gaussian case (alpha=2)
    
    References
    ----------
    .. [1] G. Samorodnitsky, M.S. Taqqu (1994) "Stable Non-Gaussian Random 
           Processes", Chapman & Hall.
    .. [2] J.M.P. Albin (1998) "A note on Rosiński's theorem"
    """
    mh = 1.0 / m
    d = H - 1.0 / alpha  # Fractional differencing parameter
    
    # Construct kernel for moving average representation
    # Kernel has two parts: recent past (t0) and distant past (t1)
    t0 = np.arange(1, m+1) / m
    t1 = np.arange(m+1, M*m+1) / m
    
    # Compute kernel values
    # For recent past: use t^d directly
    # For distant past: use difference t^d - (t-1)^d for efficiency
    A = (mh**(1.0/alpha)) * np.concatenate([t0**d, t1**d - (t1 - 1.0)**d])
    
    # Normalize to achieve desired scale C
    C_eff = C * (np.sum(np.abs(A)**alpha)**(-1.0/alpha))
    A = C_eff * A
    
    # Transform kernel to frequency domain
    Na = m * (M + N)
    A = np.fft.fft(A, n=Na)
    
    # Generate n independent realizations
    y = np.empty((n, N))
    for i in range(n):
        # Generate stable or Gaussian innovations
        if alpha < 2:
            Z = sstabrnd(alpha, 1, Na)  # Symmetric stable
        elif alpha == 2:
            Z = np.random.randn(Na)  # Gaussian
        else:
            raise ValueError("alpha must be ≤ 2")
        
        # Transform innovations to frequency domain
        Z = np.asarray(Z).reshape(-1)
        Z = np.fft.fft(Z, n=Na)
        
        # Multiply kernel and innovations in frequency domain (convolution)
        w = np.real(np.fft.ifft(Z * A, n=Na))
        
        # Extract and downsample to get desired output
        y[i, :] = w[:N*m:m]
    
    return y


def sstabrnd(alpha, beta, size):
    """
    Generate symmetric stable random variables.
    
    Implements the Chambers-Mallows-Stuck method for generating
    symmetric stable random variates (beta=1 parameterization).
    
    Parameters
    ----------
    alpha : float
        Stability parameter. Must be in (0, 2].
        Controls tail heaviness.
    beta : float
        Skewness parameter. For symmetric stable, should be 1.
        (This implementation uses beta as a scaling, not skewness)
    size : int
        Number of random variates to generate.
    
    Returns
    -------
    X : ndarray
        Array of length `size` containing stable random variates.
    
    Notes
    -----
    For symmetric stable distributions (skewness=0 in standard parameterization),
    the algorithm uses the representation:
    
    .. math::
        X = \\frac{\\sin(\\alpha U)}{\\cos(U)^{1/\\alpha}} 
            \\left(\\frac{\\cos((1-\\alpha)U)}{W}\\right)^{(1-\\alpha)/\\alpha}
    
    where U ~ Uniform(-π/2, π/2) and W ~ Exp(1).
    
    **Properties:**
    - Symmetric about 0
    - Scale parameter = 1
    - Heavier tails than Gaussian for alpha < 2
    - Infinite variance for alpha < 2
    
    Examples
    --------
    >>> import numpy as np
    >>> from stabpy.farima import sstabrnd
    >>> # Generate symmetric stable sample
    >>> np.random.seed(42)
    >>> x = sstabrnd(alpha=1.5, beta=1, size=1000)
    >>> print(f"Mean: {np.mean(x):.3f}")  # Should be ≈ 0
    >>> print(f"Sample std: {np.std(x):.3f}")  # Meaningless (infinite variance)
    >>> 
    >>> # Compare tail behavior
    >>> import matplotlib.pyplot as plt
    >>> x_stable = sstabrnd(1.5, 1, 10000)
    >>> x_normal = np.random.randn(10000)
    >>> plt.hist(x_stable, bins=100, alpha=0.5, label='Stable α=1.5', density=True)
    >>> plt.hist(x_normal, bins=100, alpha=0.5, label='Gaussian', density=True)
    >>> plt.xlim(-10, 10)
    >>> plt.legend()
    
    See Also
    --------
    fftlfsn : Uses this function for generating stable noise
    
    References
    ----------
    .. [1] J.M. Chambers, C.L. Mallows, B.W. Stuck (1976) "A method for 
           simulating stable random variables"
    """
    # Generate uniform random variable on (-π/2, π/2)
    U = np.pi * (np.random.rand(size) - 0.5)
    
    # Generate exponential random variable
    W = -np.log(np.random.rand(size))
    
    # Chambers-Mallows-Stuck transformation
    X = ((np.sin(alpha * U) / np.cos(U)**(1 / alpha)) * 
         ((np.cos((1 - alpha) * U) / W)**((1 - alpha) / alpha)))
    
    return X


def fftFarimaEst(x, p, q):
    """
    Estimate FARIMA model parameters using Whittle likelihood.
    
    Estimates the parameters of a FARIMA(p,d,q) model using the Whittle
    likelihood approximation based on the periodogram. This is a frequency-domain
    estimation method that is computationally efficient.
    
    Parameters
    ----------
    x : array_like
        Time series data.
    p : int
        Order of autoregressive (AR) component.
        Number of AR parameters to estimate.
    q : int
        Order of moving average (MA) component.
        Number of MA parameters to estimate.
    
    Returns
    -------
    Beta : ndarray
        Estimated parameters [d, phi_1, ..., phi_p, theta_1, ..., theta_q].
        - Beta[0]: fractional differencing parameter d
        - Beta[1:p+1]: AR coefficients
        - Beta[p+1:p+q+1]: MA coefficients
    
    Notes
    -----
    **FARIMA(p,d,q) model:**
    
    .. math::
        \\Phi(L)(1-L)^d X_t = \\Theta(L)\\varepsilon_t
    
    where:
    - Φ(L) = 1 + φ₁L + ... + φₚLᵖ is the AR polynomial
    - Θ(L) = 1 + θ₁L + ... + θᵩLᵩ is the MA polynomial
    - (1-L)ᵈ is the fractional differencing operator
    - εₜ is white noise
    
    **Whittle estimation:**
    
    1. Compute periodogram I(ωₖ) from FFT of data
    2. Minimize Whittle objective: ∑[log g(ωₖ; θ) + I(ωₖ)/g(ωₖ; θ)]
    3. Uses Nelder-Mead optimization (derivative-free)
    
    **Advantages:**
    - Fast: O(N log N) due to FFT
    - Consistent and asymptotically normal
    - Works well for long time series
    
    **Limitations:**
    - Can struggle with short series (N < 100)
    - May have local minima (try different initializations)
    - Less efficient than MLE but much faster
    
    Examples
    --------
    >>> import numpy as np
    >>> from stabpy.farima import fftfarima, fftFarimaEst
    >>> # Generate FARIMA(1, 0.3, 1) process
    >>> np.random.seed(42)
    >>> true_params = (2.0, 0.3, 5, [0.5], [0.3], 10, 500)
    >>> X = fftfarima(*true_params)
    >>> x = X[0, :]  # Take first realization
    >>> 
    >>> # Estimate parameters
    >>> estimated = fftFarimaEst(x, p=1, q=1)
    >>> print(f"True: d=0.3, φ=0.5, θ=0.3")
    >>> print(f"Estimated: d={estimated[0]:.3f}, φ={estimated[1]:.3f}, θ={estimated[2]:.3f}")
    >>> 
    >>> # FARIMA(0, d, 0) - only fractional differencing
    >>> X_fd = fftfarima(2.0, 0.4, 3, None, None, 10, 1000)
    >>> estimated_d = fftFarimaEst(X_fd[0, :], p=0, q=0)
    >>> print(f"True d=0.4, Estimated d={estimated_d[0]:.3f}")
    
    See Also
    --------
    IntegralEst : Objective function being minimized
    fftfarima : Generates FARIMA processes
    
    References
    ----------
    .. [1] P. Whittle (1953) "Estimation and information in stationary time series"
    .. [2] C.W.J. Granger, R. Joyeux (1980) "An introduction to long-memory 
           time series models and fractional differencing"
    .. [3] M.J. Reisen (1994) "Estimation of the fractional difference parameter"
    """
    N = len(x)
    
    # Compute FFT of data
    X = np.fft.fft(x)
    
    # Remove DC component and compute periodogram
    # Periodogram = |FFT|^2 / N
    X_no_dc = np.delete(X, 0)
    P = np.abs(X_no_dc[:N//2])**2
    
    # Frequency grid for first N/2 frequencies
    y = (2*np.pi/N) * np.arange(1, N//2 + 1)
    
    # Initial parameter guess: all zeros
    # [d, phi_1, ..., phi_p, theta_1, ..., theta_q]
    initial = np.zeros(1 + p + q)
    
    # Minimize Whittle objective using Nelder-Mead
    res = minimize(
        lambda beta: IntegralEst(beta, p, q, y, P),
        initial,
        method='Nelder-Mead'
    )
    
    return res.x


def fftfarima(alpha, d, n, Ph, Th, M, N):
    """
    Generate FARIMA process with stable innovations.
    
    Simulates realizations of a Fractionally Integrated ARMA process driven
    by symmetric stable innovations. Uses FFT-based methods for efficiency.
    
    Parameters
    ----------
    alpha : float
        Stability parameter of innovations. Must be in (0, 2].
        - alpha = 2: Gaussian FARIMA
        - alpha < 2: Heavy-tailed FARIMA
    d : float
        Fractional differencing parameter.
        - d = 0: Standard ARMA
        - 0 < d < 0.5: Stationary long memory
        - 0.5 ≤ d < 1: Non-stationary long memory
        - d = 1: ARIMA (unit root)
    n : int
        Number of independent realizations to generate.
    Ph : list or None
        AR coefficients [φ₁, φ₂, ..., φₚ].
        If None or empty, no AR component.
    Th : list or None
        MA coefficients [θ₁, θ₂, ..., θᵩ].
        If None or empty, no MA component.
    M : int
        Burn-in period (number of initial values to discard).
        Larger M gives better approximation to infinite past.
    N : int
        Length of each output realization (after burn-in).
    
    Returns
    -------
    Y : ndarray
        Array of shape (n, N) containing n independent FARIMA realizations,
        each of length N.
    
    Raises
    ------
    ValueError
        If alpha is not in (0, 2].
    
    Notes
    -----
    **FARIMA(p,d,q) with stable innovations:**
    
    .. math::
        \\Phi(L)(1-L)^d X_t = \\Theta(L)Z_t
    
    where Zₜ ~ S_α(1, 0, 0) are i.i.d. symmetric alpha-stable innovations.
    
    **Algorithm:**
    
    1. Generate fractional differencing coefficients using `fracdiff`
    2. Apply ARMA filter using `lfilter`
    3. Transform to frequency domain via FFT
    4. Generate stable (or Gaussian) innovations
    5. Multiply in frequency domain and inverse FFT
    6. Discard burn-in and return final N values
    
    **Special cases:**
    - Ph=None, Th=None, d=0: White noise
    - Ph=None, Th=None, d>0: Fractional noise
    - alpha=2, d=0: Gaussian ARMA
    - alpha=2, d>0: Gaussian FARIMA
    
    **Applications:**
    - Financial volatility modeling
    - Network traffic with long-range dependence and bursts
    - Hydrology (river flows with memory and extremes)
    
    Examples
    --------
    >>> import numpy as np
    >>> from stabpy.farima import fftfarima
    >>> # Generate Gaussian FARIMA(1,0.3,1)
    >>> np.random.seed(42)
    >>> Y_gauss = fftfarima(alpha=2.0, d=0.3, n=5, 
    ...                     Ph=[0.5], Th=[0.3], M=100, N=500)
    >>> print(f"Shape: {Y_gauss.shape}")  # (5, 500)
    >>> 
    >>> # Generate heavy-tailed FARIMA(0,0.4,0)
    >>> Y_stable = fftfarima(alpha=1.5, d=0.4, n=3,
    ...                      Ph=None, Th=None, M=50, N=1000)
    >>> 
    >>> # Visualize sample path
    >>> import matplotlib.pyplot as plt
    >>> plt.figure(figsize=(12, 4))
    >>> plt.subplot(1, 2, 1)
    >>> plt.plot(Y_gauss[0, :])
    >>> plt.title('Gaussian FARIMA(1,0.3,1)')
    >>> plt.subplot(1, 2, 2)
    >>> plt.plot(Y_stable[0, :])
    >>> plt.title('Stable FARIMA(0,0.4,0), α=1.5')
    
    See Also
    --------
    fracdiff : Computes fractional differencing coefficients
    sstabrnd : Generates stable innovations
    fftFarimaEst : Estimates FARIMA parameters
    
    References
    ----------
    .. [1] C.W.J. Granger, R. Joyeux (1980) "An introduction to long-memory 
           time series models and fractional differencing"
    .. [2] J. Beran (1994) "Statistics for Long-Memory Processes"
    .. [3] P. Kokoszka, M.S. Taqqu (1996) "Parameter estimation for infinite 
           variance fractional ARIMA"
    """
    # Generate fractional differencing coefficients
    c = fracdiff(d, M)
    
    # Handle None or empty ARMA coefficients
    if Ph is None or len(Ph) == 0:
        Ph = [0]
    if Th is None or len(Th) == 0:
        Th = [0]
    
    # Apply ARMA filter to fractional differencing coefficients
    # This combines (1-L)^d with ARMA(p,q) polynomials
    c = lfilter([1] + [-t for t in Th], [1] + [-p for p in Ph], c)
    
    # Transform filter coefficients to frequency domain
    c = fft(c, M + N)
    
    # Initialize output array
    Y = np.zeros((n, N))
    
    # Generate n independent realizations
    for i in range(n):
        # Generate innovations (stable or Gaussian)
        if alpha < 2:
            Z = sstabrnd(alpha, 1, M + N)  # Stable innovations
        elif alpha == 2:
            Z = np.random.randn(M + N)  # Gaussian innovations
        else:
            raise ValueError("alpha must be ≤ 2")
        
        # Transform innovations to frequency domain
        Z = fft(Z, M + N)
        
        # Apply filter in frequency domain (convolution in time domain)
        y = np.real(ifft(c * Z))
        
        # Discard burn-in period M and keep final N values
        Y[i, :] = y[:N]
    
    return Y
