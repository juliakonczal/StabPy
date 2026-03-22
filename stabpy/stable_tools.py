"""
Stable Distributions Toolkit
=============================

This module provides tools for working with stable (alpha-stable) distributions,
including parameter estimation, random number generation, cumulative distribution
functions, and goodness-of-fit tests.

Stable distributions are a family of probability distributions that generalize
the normal distribution and are characterized by four parameters: alpha (tail index),
beta (skewness), sigma (scale), and mu (location).

References
----------
.. [1] J.P. Nolan (1997) "Numerical calculation of stable densities and 
       distribution functions", Communications in Statistics - Stochastic 
       Models, 13(4), 759-774.
.. [2] J.M. Chambers, C.L. Mallows, B.W. Stuck (1976) "A method for simulating 
       stable random variables", Journal of the American Statistical Association, 
       71, 340-344.
.. [3] J.H. McCulloch (1986) "Simple consistent estimators of stable 
       distribution parameters", Communications in Statistics - Simulation and 
       Computation, 15(4), 1109-1136.
.. [4] I.A. Koutrouvelis (1980) "Regression-type estimation of the parameters 
       of stable laws", Journal of the American Statistical Association, 
       75(372), 918-928.
"""

import numpy as np


def stabcdf(x, alpha, sigma, beta=0.0, mu=0.0, n=2000):
    """
    Cumulative distribution function (CDF) of stable distribution.
    
    Computes the CDF of a stable distribution using numerical integration.
    The implementation handles the special case of alpha=1 (Cauchy-like)
    separately from the general case.
    
    Parameters
    ----------
    x : array_like
        Points at which to evaluate the CDF. Can be scalar or array.
    alpha : float
        Tail index (stability parameter). Must be in (0, 2].
        - alpha = 2: Gaussian distribution
        - alpha = 1: Cauchy-like distribution
        - alpha < 1: heavy-tailed distribution
    sigma : float
        Scale parameter. Must be positive.
    beta : float, optional
        Skewness parameter. Must be in [-1, 1]. Default is 0 (symmetric).
        - beta = 0: symmetric distribution
        - beta > 0: right-skewed
        - beta < 0: left-skewed
    mu : float, optional
        Location parameter. Default is 0.
    n : int, optional
        Number of integration points for numerical computation. Default is 2000.
        Higher values give more accuracy but slower computation.
    
    Returns
    -------
    y : ndarray
        CDF values at points x. Same shape as input x.
    
    Notes
    -----
    The CDF is computed using the integral representation of stable distributions.
    For alpha=1, uses a specialized formula. For other alpha values, uses
    Zolotarev's (M) parameterization and numerical integration.
    
    Examples
    --------
    >>> import numpy as np
    >>> from stable_tools import stabcdf
    >>> # Evaluate CDF at a single point
    >>> stabcdf(0, alpha=1.5, sigma=1.0, beta=0.0, mu=0.0)
    0.5
    >>> # Evaluate CDF at multiple points
    >>> x = np.array([-1, 0, 1])
    >>> stabcdf(x, alpha=1.5, sigma=1.0)
    array([0.24..., 0.5, 0.75...])
    
    References
    ----------
    .. [1] J.P. Nolan (1997) "Numerical calculation of stable densities and 
           distribution functions"
    """
    # Convert input to array and flatten
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.zeros_like(x)
    
    piby2 = 0.5 * np.pi
    
    # Special case: alpha = 1 (Cauchy-like distribution)
    if alpha == 1:
        xx = (x - mu) / sigma - beta * 2 / np.pi * np.log(sigma)
        sg = 0
        
        if beta == 0:
            # Symmetric case: standard Cauchy CDF
            y = 0.5 + (1/np.pi) * np.arctan(xx)
        else:
            # Skewed case: use numerical integration
            bb = beta
            if bb < 0:
                # Handle negative skewness by reflection
                bb = -bb
                xx = -xx
                sg = 1
            
            teta0 = np.pi * 0.5
            teta = np.arange(1, n) * (0.5*np.pi + teta0)/n - teta0
            T = teta[:, None]
            
            # Compute integrand using Zolotarev's formula
            V = (2/np.pi) * (0.5*np.pi + bb * T) / np.cos(T)
            V = V * np.exp(((0.5*np.pi + bb*T)/bb) * np.tan(T))
            G = np.exp(-0.5*np.pi * (xx[None, :]) / bb) * V
            G = np.exp(-G)
            
            # Numerical integration using trapezoidal rule
            dt = teta[1] - teta[0]
            I = np.sum(G, axis=0) * dt
            F = (1/np.pi) * I
            y = F + sg * (1 - 2*F)
    
    # General case: alpha != 1
    else:
        # Standardize and shift
        xx = (x - mu)/sigma - beta * np.tan(0.5*np.pi*alpha)
        zeta = -beta * np.tan(0.5*np.pi*alpha)
        teta0 = (1/alpha) * np.arctan(beta * np.tan(0.5*np.pi*alpha))
        xt = xx - zeta
        
        # Handle positive values of xt
        k1 = np.where(xt > 0)[0]
        if (-teta0 < 0.5*np.pi) and (k1.size > 0):
            teta = np.arange(1, n) * (0.5*np.pi + teta0)/n - teta0
            T = teta[:, None]
            
            # Compute integrand using Zolotarev's (M) parameterization
            V = np.cos(alpha*teta0 + (alpha-1)*T) / np.cos(T)
            V = V * (np.cos(T)/np.sin(alpha*(teta0+T))) ** (alpha/(alpha-1))
            V = V * (np.cos(alpha*teta0)) ** (1/(alpha-1))
            G = (xt[k1][None, :]) ** (alpha/(alpha-1))
            G = G * V
            G = np.exp(-G)
            
            # Numerical integration
            dt = teta[1] - teta[0]
            I = np.sum(G, axis=0) * dt
            c1 = (1 if alpha > 1 else 0) + (1/np.pi) * (0.5*np.pi - teta0) * (1 if alpha < 1 else 0)
            y[k1] = np.sign(1-alpha)/np.pi * I + c1
        
        # Handle zero values of xt
        k0 = np.where(xt == 0)[0]
        if k0.size > 0:
            y[k0] = (1/np.pi) * (0.5*np.pi - teta0)
        
        # Handle negative values of xt (by symmetry)
        k2 = np.where(xt < 0)[0]
        if k2.size > 0:
            teta0m = -teta0
            xt2 = -xt[k2]
            if (-teta0m < 0.5*np.pi):
                teta = np.arange(1, n) * (0.5*np.pi + teta0m)/n - teta0m
                T = teta[:, None]
                V = np.cos(alpha*teta0m + (alpha-1)*T) / np.cos(T)
                V = V * (np.cos(T)/np.sin(alpha*(teta0m+T))) ** (alpha/(alpha-1))
                V = V * (np.cos(alpha*teta0m)) ** (1/(alpha-1))
                G = (xt2[None, :]) ** (alpha/(alpha-1))
                G = G * V
                G = np.exp(-G)
                dt = teta[1] - teta[0]
                I = np.sum(G, axis=0) * dt
                c1 = (1 if alpha > 1 else 0) + (1/np.pi) * (0.5*np.pi - teta0m) * (1 if alpha < 1 else 0)
                y[k2] = 1 - np.sign(1-alpha)/np.pi * I - c1
    
    return y


def _percentile(v, p):
    """
    Compute percentile with compatibility across NumPy versions.
    
    Parameters
    ----------
    v : array_like
        Input array
    p : float or array_like
        Percentile(s) to compute
    
    Returns
    -------
    percentile : float or ndarray
        Computed percentile value(s)
    
    Notes
    -----
    NumPy changed the percentile API in version 1.22:
    - Older versions (<1.22) use 'interpolation' parameter
    - Newer versions (>=1.22) use 'method' parameter
    This function handles both cases automatically.
    """
    try:
        # Newer numpy versions (>=1.22) use 'method'
        return np.percentile(v, p, method='linear')
    except TypeError:
        # Older numpy versions (<1.22) use 'interpolation'
        return np.percentile(v, p, interpolation='linear')


def stabcull(x):
    """
    Estimate stable distribution parameters using quantile method.
    
    Implements McCulloch's (1986) quantile-based parameter estimation method.
    This is a fast, non-iterative method that uses sample quantiles and
    lookup tables to estimate the four parameters of a stable distribution.
    
    Parameters
    ----------
    x : array_like
        Sample data from which to estimate parameters. Should be 1-dimensional.
    
    Returns
    -------
    alpha : float
        Estimated tail index (stability parameter), in (0, 2].
    sigma : float
        Estimated scale parameter, positive.
    beta : float
        Estimated skewness parameter, in [-1, 1].
    mu : float
        Estimated location parameter.
    
    Notes
    -----
    The method uses the 5th, 25th, 50th, 75th, and 95th percentiles of the
    data to compute three statistics (va, vb, vs) which are then matched
    against tabulated values using bilinear interpolation.
    
    This method is:
    - Fast: no iteration or optimization required
    - Robust: based on order statistics
    - Good for initial estimates: can be refined with regression method
    
    The lookup tables (psi1, psi2, psi3, psi4) are from McCulloch (1986)
    and relate sample quantiles to distribution parameters.
    
    Examples
    --------
    >>> import numpy as np
    >>> from stable_tools import stabrnd, stabcull
    >>> # Generate stable random sample
    >>> x = stabrnd(1.5, 0.5, 1000, 1).flatten()
    >>> # Estimate parameters
    >>> alpha, sigma, beta, mu = stabcull(x)
    >>> print(f"Estimated alpha: {alpha:.3f}")
    
    References
    ----------
    .. [1] J.H. McCulloch (1986) "Simple consistent estimators of stable 
           distribution parameters", Communications in Statistics - Simulation 
           and Computation, 15(4), 1109-1136.
    """
    # Convert input to 1D array
    x = np.asarray(x, dtype=float).reshape(-1)
    x_sorted = np.sort(x)
    
    # Compute key percentiles
    x05 = _percentile(x_sorted, 5)
    x25 = _percentile(x_sorted, 25)
    x50 = _percentile(x_sorted, 50)   # median
    x75 = _percentile(x_sorted, 75)
    x95 = _percentile(x_sorted, 95)
    
    # Compute three diagnostic statistics from percentiles
    va = (x95 - x05) / (x75 - x25)      # tail weight measure
    vb = (x95 + x05 - 2*x50) / (x95 - x05)  # asymmetry measure
    vs = (x75 - x25)                     # scale measure
    
    # Lookup table grid points for va (tail weight)
    tva = np.array([2.439, 2.5, 2.6, 2.7, 2.8, 3.0, 3.2, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 25.0])
    # Lookup table grid points for vb (asymmetry)
    tvb = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    # Grid for refined alpha estimates
    ta = np.array([2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    # Grid for refined beta estimates
    tb = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    
    # Lookup table 1: maps (va, vb) to alpha
    # Rows correspond to tva values, columns to tvb values
    psi1 = np.array([
        [2.000, 2.000, 2.000, 2.000, 2.000, 2.000, 2.000],
        [1.916, 1.924, 1.924, 1.924, 1.924, 1.924, 1.924],
        [1.808, 1.813, 1.829, 1.829, 1.829, 1.829, 1.829],
        [1.729, 1.730, 1.737, 1.745, 1.745, 1.745, 1.745],
        [1.664, 1.663, 1.663, 1.668, 1.676, 1.676, 1.676],
        [1.563, 1.560, 1.553, 1.548, 1.547, 1.547, 1.547],
        [1.484, 1.480, 1.471, 1.460, 1.448, 1.438, 1.438],
        [1.391, 1.386, 1.378, 1.364, 1.337, 1.318, 1.318],
        [1.279, 1.273, 1.266, 1.250, 1.210, 1.184, 1.150],
        [1.128, 1.121, 1.114, 1.101, 1.067, 1.027, 0.973],
        [1.029, 1.021, 1.014, 1.004, 0.974, 0.935, 0.874],
        [0.896, 0.892, 0.887, 0.883, 0.855, 0.823, 0.769],
        [0.818, 0.812, 0.806, 0.801, 0.780, 0.756, 0.691],
        [0.698, 0.695, 0.692, 0.689, 0.676, 0.656, 0.595],
        [0.593, 0.590, 0.588, 0.586, 0.579, 0.563, 0.513],
    ])
    
    # Lookup table 2: maps (va, vb) to beta (unsigned)
    psi2 = np.array([
        [0.000, 2.160, 1.000, 1.000, 1.000, 1.000, 1.000],
        [0.000, 1.592, 3.390, 1.000, 1.000, 1.000, 1.000],
        [0.000, 0.759, 1.800, 1.000, 1.000, 1.000, 1.000],
        [0.000, 0.482, 1.048, 1.694, 1.000, 1.000, 1.000],
        [0.000, 0.360, 0.760, 1.232, 2.229, 1.000, 1.000],
        [0.000, 0.253, 0.518, 0.823, 1.575, 1.000, 1.000],
        [0.000, 0.203, 0.410, 0.632, 1.244, 1.906, 1.000],
        [0.000, 0.165, 0.332, 0.499, 0.943, 1.560, 1.000],
        [0.000, 0.136, 0.271, 0.404, 0.689, 1.230, 2.195],
        [0.000, 0.109, 0.216, 0.323, 0.539, 0.827, 1.917],
        [0.000, 0.096, 0.190, 0.284, 0.472, 0.693, 1.759],
        [0.000, 0.082, 0.163, 0.243, 0.412, 0.601, 1.596],
        [0.000, 0.074, 0.147, 0.220, 0.377, 0.546, 1.482],
        [0.000, 0.064, 0.128, 0.191, 0.330, 0.478, 1.362],
        [0.000, 0.056, 0.112, 0.167, 0.285, 0.428, 1.274],
    ])
    
    # Lookup table 3: maps (alpha, beta) to standardized scale
    # Rows correspond to ta values, columns to tb values
    psi3 = np.array([
        [1.908, 1.908, 1.908, 1.908, 1.908],
        [1.914, 1.915, 1.916, 1.918, 1.921],
        [1.921, 1.922, 1.927, 1.936, 1.947],
        [1.927, 1.930, 1.943, 1.961, 1.987],
        [1.933, 1.940, 1.962, 1.997, 2.043],
        [1.939, 1.952, 1.988, 2.045, 2.116],
        [1.946, 1.967, 2.022, 2.106, 2.211],
        [1.955, 1.984, 2.067, 2.188, 2.333],
        [1.965, 2.007, 2.125, 2.294, 2.491],
        [1.980, 2.040, 2.205, 2.435, 2.696],
        [2.000, 2.085, 2.311, 2.624, 2.973],
        [2.040, 2.149, 2.461, 2.886, 3.356],
        [2.098, 2.244, 2.676, 3.265, 3.912],
        [2.189, 2.392, 3.004, 3.844, 4.775],
        [2.337, 2.635, 3.542, 4.808, 6.247],
        [2.588, 3.073, 4.534, 6.636, 9.144],
    ])
    
    # Lookup table 4: maps (alpha, beta) to location adjustment
    psi4 = np.array([
        [0.0,    0.0,    0.0,    0.0,  0.0],
        [0.0, -0.017, -0.032, -0.049, -0.064],
        [0.0, -0.030, -0.061, -0.092, -0.123],
        [0.0, -0.043, -0.088, -0.132, -0.179],
        [0.0, -0.056, -0.111, -0.170, -0.232],
        [0.0, -0.066, -0.134, -0.206, -0.283],
        [0.0, -0.075, -0.154, -0.241, -0.335],
        [0.0, -0.084, -0.173, -0.276, -0.390],
        [0.0, -0.090, -0.192, -0.310, -0.447],
        [0.0, -0.095, -0.208, -0.346, -0.508],
        [0.0, -0.098, -0.223, -0.383, -0.576],
        [0.0, -0.099, -0.237, -0.424, -0.652],
        [0.0, -0.096, -0.250, -0.469, -0.742],
        [0.0, -0.089, -0.262, -0.520, -0.853],
        [0.0, -0.078, -0.272, -0.581, -0.997],
        [0.0, -0.061, -0.279, -0.659, -1.198],
    ])
    
    # Find bracketing indices for bilinear interpolation in (va, vb) space
    tvai1 = max(0, np.max(np.where(tva <= va)[0], initial=0))
    tvai2 = min(14, np.min(np.where(tva >= va)[0], initial=14))
    tvbi1 = max(0, np.max(np.where(tvb <= abs(vb))[0], initial=0))
    tvbi2 = min(6, np.min(np.where(tvb >= abs(vb))[0], initial=6))
    
    # Compute interpolation weights
    dista = tva[tvai2] - tva[tvai1]
    dista = (va - tva[tvai1]) / dista if dista != 0 else 0.0
    distb = tvb[tvbi2] - tvb[tvbi1]
    distb = (abs(vb) - tvb[tvbi1]) / distb if distb != 0 else 0.0
    
    # Bilinear interpolation to estimate alpha from psi1 table
    psi1b1 = dista * psi1[tvai2, tvbi1] + (1-dista) * psi1[tvai1, tvbi1]
    psi1b2 = dista * psi1[tvai2, tvbi2] + (1-dista) * psi1[tvai1, tvbi2]
    alpha = distb * psi1b2 + (1-distb) * psi1b1
    
    # Bilinear interpolation to estimate beta (unsigned) from psi2 table
    psi2b1 = dista * psi2[tvai2, tvbi1] + (1-dista) * psi2[tvai1, tvbi1]
    psi2b2 = dista * psi2[tvai2, tvbi2] + (1-dista) * psi2[tvai1, tvbi2]
    beta = np.sign(vb) * (distb * psi2b2 + (1 - distb) * psi2b1)  # Apply sign from vb
    
    # Find bracketing indices in (alpha, beta) space for scale/location estimation
    tai1 = max(0, np.max(np.where(ta >= alpha)[0], initial=0))
    tai2 = min(15, np.min(np.where(ta <= alpha)[0], initial=15))
    tbi1 = max(0, np.max(np.where(tb <= abs(beta))[0], initial=0))
    tbi2 = min(4, np.min(np.where(tb >= abs(beta))[0], initial=4))
    
    # Compute interpolation weights
    dista = ta[tai2] - ta[tai1]
    dista = (alpha - ta[tai1]) / dista if dista != 0 else 0.0
    distb = tb[tbi2] - tb[tbi1]
    distb = (abs(beta) - tb[tbi1]) / distb if distb != 0 else 0.0
    
    # Estimate sigma using psi3 table
    psi3b1 = dista * psi3[tai2, tbi1] + (1-dista) * psi3[tai1, tbi1]
    psi3b2 = dista * psi3[tai2, tbi2] + (1-dista) * psi3[tai1, tbi2]
    sigma = vs / (distb * psi3b2 + (1 - distb) * psi3b1)
    
    # Estimate location using psi4 table
    psi4b1 = dista * psi4[tai2, tbi1] + (1-dista) * psi4[tai1, tbi1]
    psi4b2 = dista * psi4[tai2, tbi2] + (1-dista) * psi4[tai1, tbi2]
    zeta = np.sign(beta) * sigma * (distb * psi4b2 + (1 - distb) * psi4b1) + x50
    
    # Convert zeta to mu depending on alpha
    if abs(alpha - 1) < 0.05:
        # Near alpha=1, use zeta directly
        mu = zeta
    else:
        # Standard conversion formula
        mu = zeta - beta * sigma * np.tan(0.5 * np.pi * alpha)
    
    # Enforce parameter constraints
    if alpha <= 0: alpha = 1e-10
    if alpha > 2: alpha = 2.0
    if sigma <= 0: sigma = 1e-10
    beta = min(1.0, max(-1.0, beta))
    
    return float(alpha), float(sigma), float(beta), float(mu)


def stabreg(x, epsilon=1e-5, maxit=5, deltac=None):
    """
    Estimate stable distribution parameters using regression method.
    
    Implements Koutrouvelis's (1980) regression-based parameter estimation method.
    This iterative method refines parameter estimates using the empirical
    characteristic function and is generally more accurate than the quantile method,
    though slower.
    
    Parameters
    ----------
    x : array_like
        Sample data from which to estimate parameters.
    epsilon : float, optional
        Convergence tolerance for the scale correction factor. Default is 1e-5.
    maxit : int, optional
        Maximum number of iterations. Default is 5.
    deltac : array_like, optional
        Array of candidate values for the location shift parameter.
        If None, uses np.arange(100). Default is None.
    
    Returns
    -------
    alpha : float
        Estimated tail index (stability parameter).
    sigma : float
        Estimated scale parameter.
    beta : float
        Estimated skewness parameter.
    mu : float
        Estimated location parameter.
    
    Notes
    -----
    The method works by:
    1. Getting initial estimates from stabcull (quantile method)
    2. Standardizing the data
    3. Using the empirical characteristic function to estimate alpha
    4. Iteratively refining scale and location parameters
    5. Using regression on log-characteristic function for beta estimation
    
    The algorithm uses optimal values of K and L (number of frequency points)
    that depend on the estimated alpha, as determined by Koutrouvelis.
    
    This method typically provides more accurate estimates than stabcull,
    especially for larger sample sizes, but requires more computation time.
    
    Examples
    --------
    >>> import numpy as np
    >>> from stable_tools import stabrnd, stabreg
    >>> # Generate stable random sample
    >>> x = stabrnd(1.5, 0.5, 1000, 1).flatten()
    >>> # Estimate parameters using regression method
    >>> alpha, sigma, beta, mu = stabreg(x)
    >>> print(f"Estimated parameters: alpha={alpha:.3f}, beta={beta:.3f}")
    
    References
    ----------
    .. [1] I.A. Koutrouvelis (1980) "Regression-type estimation of the parameters 
           of stable laws", Journal of the American Statistical Association, 
           75(372), 918-928.
    """
    # Convert to column vector
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    n = x.shape[0]
    
    # Setup candidate location shifts
    if deltac is None:
        delc = np.arange(100)
    else:
        delc = np.asarray(deltac, dtype=int)
    
    # Optimal frequency points for different alpha ranges (from Koutrouvelis)
    Kopt   = np.array([9, 11, 16, 18, 22, 24, 68, 124])  # for alpha estimation
    Lopt   = np.array([10, 14, 16, 18, 14, 16, 38, 68])  # for beta estimation
    indexA = np.array([1.9, 1.5, 1.3, 1.1, 0.9, 0.7, 0.5, 0.3])  # alpha thresholds
    
    # Get initial parameter estimates using quantile method
    alpha, sigma, beta, mu = stabcull(x)
    
    # Standardize data
    X = (x - mu) / sigma
    
    def _alpha1_from_X(X, K):
        """
        Estimate alpha using empirical characteristic function.
        
        Uses regression on log-log plot of characteristic function modulus.
        """
        t = np.arange(1, K+1) * np.pi / 25.0
        w = np.log(np.abs(t))
        w1 = w - np.mean(w)
        yvals = []
        for tt in t:
            yvals.append(np.mean(np.exp(1j * tt * X)))
        y = np.log(-2 * np.log(np.abs(yvals)))
        alpha1 = np.sum(w1 * (y - np.mean(y))) / np.sum(w1 * w1)
        return float(np.real(alpha1))
    
    # Initial alpha estimation
    K = 11
    alpha1 = _alpha1_from_X(X, K)
    if alpha1 <= 0.9:
        # For small alpha, use more frequency points
        K = 30
        alpha1 = _alpha1_from_X(X, K)
    
    # Initialize iteration variables
    it = 1
    c1 = 0.0  # scale correction factor
    beta2 = beta
    delta2 = 0.0  # location shift
    
    def _choose_idx(alpha1):
        """Select optimal K and L based on current alpha estimate."""
        idx = np.where(indexA <= alpha1)[0]
        return min(idx[0] if idx.size > 0 else 7, 7)
    
    # Iterative refinement
    while (it <= maxit) and (abs(c1 - 1) > epsilon):
        # Select optimal frequency points based on current alpha
        idx = _choose_idx(alpha1)
        K = int(Kopt[idx])
        t = np.arange(1, K+1) * np.pi / 25.0
        w = np.log(np.abs(t))
        w1 = w - np.mean(w)
        
        # Compute empirical characteristic function
        yvals = []
        for tt in t:
            yvals.append(np.mean(np.exp(1j * tt * X)))
        y = np.log(-2 * np.log(np.abs(yvals)))
        
        # Refine alpha estimate via regression
        alpha1 = np.sum(w1 * (y - np.mean(y))) / np.sum(w1 * w1)
        alpha1 = float(np.real(alpha1))
        
        # Compute scale correction factor
        c1 = (0.5 * np.exp(np.mean(y - alpha1 * w))) ** (1/alpha1)
        
        # Setup for beta estimation
        L = int(Lopt[idx])
        u = np.arange(1, L+1) * np.pi / 50.0
        
        # Rescale data
        X = X / c1
        
        # Compute sums for characteristic function phase
        sinXu = np.array([np.sum(np.sin(uu * X)) for uu in u], dtype=float)
        cosXu = np.array([np.sum(np.cos(uu * X)) for uu in u], dtype=float)
        
        # Find optimal location shift to maximize symmetry
        testcos = ((np.ones((delc.size, 1)) * cosXu) * np.cos(delc[:, None] * u) +
                   (np.ones((delc.size, 1)) * sinXu) * np.sin(delc[:, None] * u))
        testcos = np.sum(np.abs(np.diff(np.sign(testcos.T), axis=0)), axis=0)
        idx0 = np.where(testcos == 0)[0]
        if idx0.size == 0:
            break
        deltac = delc[int(idx0.min())]
        
        # Apply location shift
        X = X - deltac
        
        # Estimate beta using regression on characteristic function phase
        num = (sinXu * np.cos(deltac) - cosXu * np.sin(deltac))
        den = (cosXu * np.cos(deltac) + sinXu * np.sin(deltac))
        z = np.arctan2(num, den)
        yvec = (c1 ** alpha1) * np.tan(np.pi * alpha1 / 2) * np.sign(u) * (u ** alpha1)
        
        # Solve for beta and delta using least squares
        S_yy = np.sum(yvec * yvec)
        S_uu = np.sum(u * u)
        S_uy = np.sum(u * yvec)
        S_uz = np.sum(u * z)
        S_yz = np.sum(yvec * z)
        
        denom = S_uu * S_yy - (S_uy ** 2)
        if denom == 0:
            break
        delta2 = (S_yy * S_uz - S_uy * S_yz) / denom
        beta2 = (S_uu * S_yz - S_uy * S_uz) / denom
        
        # Update scale and location
        sigma = sigma * c1
        mu = mu + deltac * sigma
        it += 1
    
    # Final parameter values
    alpha = alpha1
    beta = beta2
    mu = mu + sigma * delta2
    
    # Enforce parameter constraints
    alpha = max(min(alpha, 2.0), 1e-10)
    sigma = max(sigma, 1e-10)
    beta = min(1.0, max(-1.0, beta))
    
    return float(alpha), float(sigma), float(beta), float(mu)


def stabrnd(alpha, beta, m, n):
    """
    Generate random numbers from stable distribution.
    
    Implements Chambers-Mallows-Stuck (1976) method for generating
    stable random variates using uniform and exponential random numbers.
    
    Parameters
    ----------
    alpha : float
        Tail index (stability parameter). Must be in (0, 2].
    beta : float
        Skewness parameter. Must be in [-1, 1].
    m : int
        Number of rows in output array.
    n : int
        Number of columns in output array.
    
    Returns
    -------
    r : ndarray
        Array of shape (m, n) containing stable random variates with
        parameters (alpha, beta, 0, 1) - i.e., standard stable distribution.
        To get general stable variates, use: sigma * r + mu
    
    Notes
    -----
    The generated values follow the standard stable distribution (sigma=1, mu=0).
    The method uses the Chambers-Mallows-Stuck algorithm which is based on
    transforming uniform and exponential random variables.
    
    Special cases:
    - alpha=2: Normal distribution (use sigma * np.random.randn(m,n) + mu instead)
    - alpha=1, beta=0: Cauchy distribution
    
    Invalid parameters (alpha not in (0,2] or |beta|>1) return NaN arrays.
    
    Examples
    --------
    >>> import numpy as np
    >>> from stable_tools import stabrnd
    >>> # Generate standard stable random numbers
    >>> x = stabrnd(1.5, 0.5, 100, 1)
    >>> # Generate general stable variates
    >>> sigma, mu = 2.0, 1.0
    >>> y = sigma * stabrnd(1.5, 0.5, 100, 1) + mu
    
    References
    ----------
    .. [1] J.M. Chambers, C.L. Mallows, B.W. Stuck (1976) "A method for 
           simulating stable random variables", Journal of the American 
           Statistical Association, 71, 340-344.
    """
    alpha = float(alpha)
    beta = float(beta)
    r = np.zeros((m, n), dtype=float)
    piby2 = 1.5707963268  # π/2
    
    # Special case: alpha = 1 (Cauchy-like)
    if alpha == 1:
        V = np.pi * (np.random.rand(m, n) - 0.5)
        W = -np.log(np.random.rand(m, n))
        r = ((piby2 + beta * V) * np.tan(V) - 
             beta * np.log(piby2 * W * np.cos(V) / (piby2 + beta * V))) / piby2
    
    # General case: alpha != 1
    else:
        # Precompute constants
        tan_a = beta * np.tan(piby2 * alpha)
        B_ab = np.arctan(tan_a) / alpha
        S_ab = (1 + tan_a ** 2) ** (0.5 / alpha)
        
        # Generate uniform and exponential random variables
        V = np.pi * (np.random.rand(m, n) - 0.5)
        W = -np.log(np.random.rand(m, n))
        
        # Chambers-Mallows-Stuck transformation
        r = (S_ab * np.sin(alpha * (V + B_ab)) / (np.cos(V) ** (1 / alpha)) * 
             (np.cos(V - alpha * (V + B_ab)) / W) ** ((1 - alpha) / alpha))
    
    # Handle invalid parameter values
    if (alpha <= 0) or (alpha > 2):
        r[:] = np.nan
    if (abs(beta) > 1):
        r[:] = np.nan
    
    return r


def stabtest(x, ilp):
    """
    Goodness-of-fit test for stable distribution.
    
    Performs multiple goodness-of-fit tests (Kolmogorov-Smirnov, Kuiper,
    Cramér-von Mises, Watson, Anderson-Darling) for the stable distribution
    hypothesis. P-values are computed via Monte Carlo simulation.
    
    Parameters
    ----------
    x : array_like
        Sample data to test.
    ilp : int
        Number of Monte Carlo iterations for p-value estimation.
        Larger values give more accurate p-values but take longer.
        Recommended: at least 1000.
    
    Returns
    -------
    y : ndarray
        Array of length 14 containing:
        - y[0:4]: Estimated parameters [alpha, sigma, beta, mu]
        - y[4:9]: Test statistics [D, V, W2, U2, A2] where:
            * D: Kolmogorov-Smirnov statistic
            * V: Kuiper statistic
            * W2: Cramér-von Mises statistic
            * U2: Watson statistic
            * A2: Anderson-Darling statistic
        - y[9:14]: Corresponding p-values for each test statistic
    
    Notes
    -----
    The test procedure:
    1. Estimates parameters using stabreg (regression method)
    2. Computes empirical CDF using stabcdf
    3. Calculates five test statistics comparing empirical to theoretical CDF
    4. Generates ilp random samples from estimated distribution
    5. Computes test statistics for each simulated sample
    6. P-value = proportion of simulated statistics exceeding observed value
    
    Interpretation:
    - Small p-values (< 0.05) suggest the data is not from a stable distribution
    - Large p-values indicate the stable distribution fits the data well
    - Different tests have different sensitivities to various departures from
      the null hypothesis
    
    The test statistics measure different aspects of fit:
    - D and V: focus on maximum deviations
    - W2 and U2: integrate squared deviations (U2 is location-invariant)
    - A2: gives more weight to tail deviations
    
    Examples
    --------
    >>> import numpy as np
    >>> from stable_tools import stabrnd, stabtest
    >>> # Generate stable random sample
    >>> x = stabrnd(1.5, 0.5, 500, 1).flatten() * 2 + 1
    >>> # Test goodness of fit
    >>> results = stabtest(x, ilp=1000)
    >>> print(f"Estimated alpha: {results[0]:.3f}")
    >>> print(f"KS test p-value: {results[9]:.3f}")
    >>> print(f"AD test p-value: {results[13]:.3f}")
    
    References
    ----------
    .. [1] M.A. Stephens (1974) "EDF Statistics for Goodness of Fit and Some 
           Comparisons", Journal of the American Statistical Association, 69(347), 
           730-737.
    .. [2] R.B. D'Agostino and M.A. Stephens (1986) "Goodness-of-Fit Techniques",
           Marcel Dekker, New York.
    """
    # Convert input to 1D array
    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.size
    
    # Estimate parameters using regression method
    alpha, sigma, beta, mu = stabreg(x)
    
    # Initialize result array
    y = np.zeros(14, dtype=float)
    
    # Compute sorted empirical CDF values
    z = np.sort(stabcdf(x, alpha, sigma, beta, mu))
    
    # Kolmogorov-Smirnov statistic: maximum distance between empirical and theoretical CDF
    kp = np.max(np.arange(1, n+1)/n - z)  # maximum positive deviation
    km = np.max(z - np.arange(0, n)/n)    # maximum negative deviation
    Dp = np.max([kp, km]) * (np.sqrt(n) + 0.12 + 0.11/np.sqrt(n))
    
    # Kuiper statistic: sum of maximum positive and negative deviations
    Vp = (kp + km) * (np.sqrt(n) + 0.155 + 0.24/np.sqrt(n))
    
    # Cramér-von Mises statistic: integrated squared distance
    W2 = np.sum((z - (np.arange(1, 2*n, 2)/(2*n)))**2) + 1/(12*n)
    
    # Watson statistic: Cramér-von Mises with location correction
    U2 = W2 - n * (np.mean(z) - 0.5)**2
    
    # Modified statistics with finite-sample corrections
    W2p = (W2 - 0.4/n + 0.6/n**2) * (1 + 1/n)
    U2p = (U2 - 0.1/n + 0.1/n**2) * (1 + 0.8/n)
    
    # Anderson-Darling statistic: weighted squared distance (more weight in tails)
    A2p = - (1/n) * np.dot(np.arange(1, 2*n, 2), 
                            (np.log(z) + np.log(1 - z[::-1]))) - n
    
    # Monte Carlo simulation for p-value estimation
    proba = sigma * stabrnd(alpha, beta, n, ilp) + mu
    probat = np.zeros((ilp, 5), dtype=float)
    
    for i in range(ilp):
        # Estimate parameters for simulated sample
        a_i, s_i, b_i, m_i = stabreg(proba[:, i])
        z_i = np.sort(stabcdf(proba[:, i], a_i, s_i, b_i, m_i))
        
        # Compute test statistics for simulated sample
        kp = np.max(np.arange(1, n+1)/n - z_i)
        km = np.max(z_i - np.arange(0, n)/n)
        probat[i, 0] = np.max([kp, km]) * (np.sqrt(n) + 0.12 + 0.11/np.sqrt(n))
        probat[i, 1] = (kp + km) * (np.sqrt(n) + 0.155 + 0.24/np.sqrt(n))
        
        W2 = np.sum((z_i - (np.arange(1, 2*n, 2)/(2*n)))**2) + 1/(12*n)
        U2 = W2 - n * (np.mean(z_i) - 0.5)**2
        probat[i, 2] = (W2 - 0.4/n + 0.6/n**2) * (1 + 1/n)
        probat[i, 3] = (U2 - 0.1/n + 0.1/n**2) * (1 + 0.8/n)
        probat[i, 4] = - (1/n) * np.dot(np.arange(1, 2*n, 2), 
                                         (np.log(z_i) + np.log(1 - z_i[::-1]))) - n
    
    # Compile results
    y[0:4] = [alpha, sigma, beta, mu]
    pt = np.array([Dp, Vp, W2p, U2p, A2p], dtype=float)
    y[4:9] = pt
    
    # Compute p-values: proportion of simulated statistics exceeding observed
    pvals = np.array([(probat[:, j] > pt[j]).sum()/ilp for j in range(5)], dtype=float)
    y[9:14] = pvals
    
    return y
