"""
Normal Inverse Gaussian Distribution
=====================================

This module provides tools for working with the Normal Inverse Gaussian (NIG) 
distribution, including probability density and cumulative distribution functions,
parameter estimation via maximum likelihood, random number generation, and 
goodness-of-fit tests.

The NIG distribution is a member of the generalized hyperbolic family and is
commonly used in financial modeling due to its ability to capture heavy tails
and skewness observed in asset returns.

References
----------
.. [1] O. Barndorff-Nielsen (1997) "Normal Inverse Gaussian Distributions and 
       Stochastic Volatility Modelling", Scandinavian Journal of Statistics, 
       24(1), 1-13.
.. [2] R. Weron (2004) "Computationally intensive Value at Risk calculations", 
       in "Handbook of Computational Statistics: Concepts and Methods", eds. 
       J.E. Gentle, W. Haerdle, Y. Mori, Springer, Berlin, 911-950.
.. [3] R. Weron (2007) "Modeling and Forecasting Electricity Loads and Prices: 
       A Statistical Approach", Wiley, Chichester.
.. [4] K. Prause (1999) "The Generalized Hyperbolic Model: Estimation, Financial 
       Derivatives, and Risk Measures", PhD Thesis, University of Freiburg.
"""

import numpy as np
import scipy.stats as st
from scipy.optimize import minimize
from scipy.special import kv as besselk
from scipy.integrate import quad


def nigpdf(x, alpha, beta, delta, mu):
    """
    Probability density function of the Normal Inverse Gaussian distribution.
    
    The NIG distribution is a continuous probability distribution that arises
    as a normal variance-mean mixture where the mixing density is inverse Gaussian.
    It is capable of modeling skewness and heavy tails.
    
    Parameters
    ----------
    x : array_like
        Points at which to evaluate the PDF.
    alpha : float
        Tail heaviness parameter. Must satisfy alpha > |beta|.
        Larger alpha means lighter tails.
    beta : float
        Asymmetry/skewness parameter. Must satisfy |beta| < alpha.
        - beta = 0: symmetric distribution
        - beta > 0: right-skewed
        - beta < 0: left-skewed
    delta : float
        Scale parameter. Must be positive.
        Controls the spread of the distribution.
    mu : float
        Location parameter.
        Shifts the distribution along the x-axis.
    
    Returns
    -------
    y : ndarray
        PDF values at points x.
    
    Notes
    -----
    The PDF of NIG(alpha, beta, delta, mu) is given by:
    
    .. math::
        f(x) = \\frac{\\alpha \\delta}{\\pi} 
               \\frac{K_1(\\alpha \\sqrt{\\delta^2 + (x-\\mu)^2})}{\\sqrt{\\delta^2 + (x-\\mu)^2}}
               \\exp(\\delta\\sqrt{\\alpha^2 - \\beta^2} + \\beta(x - \\mu))
    
    where K_1 is the modified Bessel function of the second kind with index 1.
    
    **Parameter constraints:**
    - delta > 0
    - alpha > |beta| (ensures positive variance)
    
    **Special cases:**
    - When beta = 0, the distribution is symmetric
    - As alpha → ∞, the distribution approaches normal
    
    If parameter constraints are violated, the function returns infinity.
    
    Examples
    --------
    >>> import numpy as np
    >>> from stabpy.nig import nigpdf
    >>> # Symmetric NIG distribution
    >>> x = np.linspace(-5, 5, 100)
    >>> pdf = nigpdf(x, alpha=1.5, beta=0.0, delta=1.0, mu=0.0)
    >>> # Right-skewed NIG distribution
    >>> pdf_skewed = nigpdf(x, alpha=1.5, beta=0.5, delta=1.0, mu=0.0)
    
    See Also
    --------
    nigcdf : Cumulative distribution function
    nigrnd : Random number generation
    
    References
    ----------
    .. [1] R. Weron (2004) "Computationally intensive Value at Risk calculations"
    .. [2] O. Barndorff-Nielsen (1997) "Normal Inverse Gaussian Distributions 
           and Stochastic Volatility Modelling"
    """
    # Lambda parameter for NIG is fixed at -1/2
    # This corresponds to the standard NIG parameterization
    lambda_ = -1/2
    
    # Check parameter constraints
    if delta > 0 and alpha > abs(beta):
        # Compute normalization constant kappa
        # Uses modified Bessel function of second kind
        gamma_param = np.sqrt(alpha**2 - beta**2)
        kappa = ((alpha**2 - beta**2)**(lambda_/2)) / (
            np.sqrt(2 * np.pi) * alpha**(lambda_ - 0.5) * delta**lambda_ * 
            besselk(lambda_, delta * gamma_param)
        )
        
        # Compute PDF value
        # Formula involves Bessel function K_{lambda-1/2} and exponential term
        r = np.sqrt(delta**2 + (x - mu)**2)
        y = (kappa * 
             (delta**2 + (x - mu)**2)**((lambda_ - 0.5) / 2) * 
             besselk(lambda_ - 0.5, alpha * r) * 
             np.exp(beta * (x - mu)))
    else:
        # Invalid parameters: return infinity
        y = np.inf * np.ones_like(x)
    
    return y


def nigloglik(params, x):
    """
    Negative log-likelihood function for NIG distribution.
    
    This function is used for maximum likelihood estimation of NIG parameters.
    The location parameter mu is derived from the other parameters and the
    sample mean, reducing the optimization to three parameters.
    
    Parameters
    ----------
    params : list or array
        Parameters [alpha, beta, delta] to optimize.
    x : array_like
        Sample data.
    
    Returns
    -------
    nll : float
        Negative log-likelihood value. Returns a very large number if
        parameter constraints are violated.
    
    Notes
    -----
    The location parameter mu is computed as:
    
    .. math::
        \\mu = -\\delta\\beta/\\gamma \\cdot K_1(\\delta\\gamma)/K_{1/2}(\\delta\\gamma) + \\bar{x}
    
    where gamma = sqrt(alpha^2 - beta^2) and K_v is the modified Bessel function
    of the second kind.
    
    This parameterization ensures that E[X] = mu, which helps with numerical
    stability during optimization.
    
    **Optimization approach:**
    This function is minimized (negative log-likelihood) to find maximum
    likelihood estimates. Used internally by `nigest`.
    
    Examples
    --------
    >>> import numpy as np
    >>> from stabpy.nig import nigloglik
    >>> x = np.random.randn(100)
    >>> # Evaluate negative log-likelihood
    >>> nll = nigloglik([1.5, 0.5, 1.0], x)
    
    See Also
    --------
    nigest : Parameter estimation using this likelihood function
    nigpdf : Probability density function
    
    References
    ----------
    .. [1] R. Weron (2004) "Computationally intensive Value at Risk calculations"
    """
    alpha = params[0]
    beta = params[1]
    delta = params[2]
    
    # Compute mu from other parameters and sample mean
    # This ensures E[X] = mu, which aids numerical stability
    gamma_param = np.sqrt(alpha**2 - beta**2)
    bessel_ratio = besselk(0.5, delta * gamma_param) / besselk(-0.5, delta * gamma_param)
    mu = -delta * beta / gamma_param * bessel_ratio + np.mean(x)
    
    # Check parameter constraints
    if delta > 0 and alpha > abs(beta):
        # Compute negative log-likelihood
        # = -sum(log(pdf(x_i)))
        pdf_vals = nigpdf(x, alpha, beta, delta, mu)
        y = -np.sum(np.log(pdf_vals))
    else:
        # Invalid parameters: return large value to prevent optimizer from going there
        y = np.finfo(float).max
    
    return y


def nigest(x, x0=None):
    """
    Estimate parameters of the Normal Inverse Gaussian distribution via MLE.
    
    Uses the Nelder-Mead simplex algorithm to maximize the likelihood function.
    This is a robust derivative-free optimization method suitable for the
    NIG distribution's complex likelihood surface.
    
    Parameters
    ----------
    x : array_like
        Sample data from which to estimate parameters.
    x0 : list or array, optional
        Initial parameter estimates [alpha, beta, delta].
        If None, uses default starting values [0.5, 0, 1].
        Good initial values can significantly speed up convergence.
    
    Returns
    -------
    params : ndarray
        Maximum likelihood estimates [alpha, beta, delta, mu].
        - alpha: tail heaviness parameter
        - beta: asymmetry parameter
        - delta: scale parameter
        - mu: location parameter
    
    Notes
    -----
    The optimization minimizes the negative log-likelihood function using
    scipy.optimize.minimize with the Nelder-Mead method.
    
    **Computational considerations:**
    - MLE for NIG can be computationally intensive for large samples
    - The likelihood surface may have local maxima
    - If convergence fails, try different initial values
    - For very skewed or heavy-tailed data, convergence may be slow
    
    **Choosing initial values:**
    - alpha ≈ 0.5-2: typical range for financial data
    - beta ≈ 0: start with symmetric case
    - delta ≈ 1: reasonable scale
    
    The location parameter mu is computed from the other three parameters
    and the sample mean to ensure E[X] = mu.
    
    Examples
    --------
    >>> import numpy as np
    >>> from stabpy.nig import nigrnd, nigest
    >>> # Generate NIG random sample
    >>> np.random.seed(42)
    >>> true_params = (1.5, 0.5, 1.0, 0.0)
    >>> x = nigrnd(*true_params, 500, 1).flatten()
    >>> # Estimate parameters
    >>> estimated = nigest(x)
    >>> print(f"True alpha: 1.5, Estimated: {estimated[0]:.3f}")
    >>> print(f"True beta: 0.5, Estimated: {estimated[1]:.3f}")
    >>> 
    >>> # Using custom initial values
    >>> estimated2 = nigest(x, x0=[2.0, 0.0, 1.5])
    
    See Also
    --------
    nigloglik : Log-likelihood function being optimized
    nigtest : Goodness-of-fit test using these estimates
    
    References
    ----------
    .. [1] R. Weron (2004) "Computationally intensive Value at Risk calculations"
    .. [2] J.A. Nelder, R. Mead (1965) "A simplex method for function minimization"
    """
    # Set default initial values if not provided
    if x0 is None:
        x0 = [0.5, 0, 1]  # [alpha, beta, delta]
    
    # Minimize negative log-likelihood using Nelder-Mead
    # This is a robust derivative-free method
    result = minimize(
        nigloglik, 
        x0, 
        args=(x,), 
        method='Nelder-Mead',
        options={'maxiter': int(1e12), 'disp': False}
    )
    
    # Extract optimization results
    params = result.x  # [alpha, beta, delta]
    fval = result.fun  # minimum negative log-likelihood
    exitflag = 0 if result.success else 1
    iterations = result.nit
    
    # Extract parameters
    alpha, beta, delta = params
    
    # Compute location parameter mu
    gamma_param = np.sqrt(alpha**2 - beta**2)
    bessel_ratio = besselk(0.5, delta * gamma_param) / besselk(-0.5, delta * gamma_param)
    mu = -delta * beta / gamma_param * bessel_ratio + np.mean(x)
    
    # Append mu to parameter vector
    params = np.append(params, mu)
    
    return params


def nigcdf(x, alpha, beta, delta, mu, starti=None):
    """
    Cumulative distribution function of the Normal Inverse Gaussian distribution.
    
    Computes the CDF by numerical integration of the PDF. The integration is
    performed from a suitable starting point (where PDF ≈ 0) up to each value in x.
    
    Parameters
    ----------
    x : array_like
        Points at which to evaluate the CDF.
    alpha : float
        Tail heaviness parameter. Must satisfy alpha > |beta|.
    beta : float
        Asymmetry parameter. Must satisfy |beta| < alpha.
    delta : float
        Scale parameter. Must be positive.
    mu : float
        Location parameter.
    starti : float, optional
        Starting point for numerical integration. If None, automatically
        determined as a point where PDF is negligible (< 1e-10).
        For most applications, leaving this as None is recommended.
    
    Returns
    -------
    y : ndarray
        CDF values at points x, in the same order as input x.
    
    Notes
    -----
    The CDF is computed by integrating the PDF:
    
    .. math::
        F(x) = \\int_{-\\infty}^x f(t) dt
    
    **Numerical integration:**
    - Uses scipy.integrate.quad for accurate quadrature
    - Automatically finds a suitable lower integration limit where PDF ≈ 0
    - Integrates cumulatively for efficiency when evaluating multiple points
    
    **Computational cost:**
    This function can be slow for large arrays due to numerical integration.
    For very large datasets, consider using approximations or sampling methods.
    
    **Accuracy:**
    The default tolerance of quad provides high accuracy (typically 8+ decimal places).
    If higher accuracy is needed, modify the quad call to include epsabs/epsrel.
    
    Examples
    --------
    >>> import numpy as np
    >>> from stabpy.nig import nigcdf, nigrnd
    >>> # Evaluate CDF at specific points
    >>> x = np.array([-2, -1, 0, 1, 2])
    >>> cdf = nigcdf(x, alpha=1.5, beta=0.5, delta=1.0, mu=0.0)
    >>> print(cdf)
    >>> 
    >>> # Verify using random sample
    >>> sample = nigrnd(1.5, 0.5, 0.0, 1.0, 1000, 1).flatten()
    >>> empirical_cdf = np.mean(sample <= 0)  # P(X <= 0)
    >>> theoretical_cdf = nigcdf(0, 1.5, 0.5, 1.0, 0.0)
    >>> print(f"Empirical: {empirical_cdf:.3f}, Theoretical: {theoretical_cdf:.3f}")
    
    See Also
    --------
    nigpdf : Probability density function being integrated
    nigtest : Uses this function for goodness-of-fit testing
    
    References
    ----------
    .. [1] R. Weron (2004) "Computationally intensive Value at Risk calculations"
    """
    # Convert input to flat array
    x = np.array(x).flatten()
    n = len(x)
    
    # Threshold for determining where PDF is negligible
    feps = 1e-10
    
    # Determine starting point for integration if not provided
    if starti is None:
        # Start near the distribution mean, accounting for skewness
        # The mean of NIG is mu + delta*beta/sqrt(alpha^2 - beta^2) * K_1/K_0
        gamma_param = np.sqrt(alpha**2 - beta**2)
        bessel_ratio = besselk(1.5, delta * gamma_param) / besselk(0.5, delta * gamma_param)
        mean_estimate = mu + beta * delta / gamma_param * bessel_ratio
        
        # Move left until PDF becomes negligible
        starti = min(mean_estimate, np.min(x)) - 1
        while nigpdf(starti, alpha, beta, delta, mu) > feps:
            starti -= 1
    
    # Initialize output array
    y = np.zeros(n)
    
    # Sort x values for efficient cumulative integration
    x_sorted_indices = np.argsort(x)
    x_sorted = np.sort(x)
    
    # Add starting point to beginning of sorted array
    x_sorted = np.concatenate(([starti], x_sorted))
    
    # Integrate PDF from starti to each x value
    # Cumulative approach: integrate intervals and accumulate
    for i in range(n):
        # Integrate from x_sorted[i] to x_sorted[i+1]
        y[i], _ = quad(nigpdf, x_sorted[i], x_sorted[i + 1], 
                       args=(alpha, beta, delta, mu))
    
    # Cumulative sum gives CDF values
    y = np.cumsum(y)
    
    # Reorder to match original input order
    y = y[np.argsort(x_sorted_indices)]
    
    return y


def invgrnd(delta, gamma, M, N):
    """
    Generate random numbers from Inverse Gaussian distribution.
    
    The Inverse Gaussian (IG) distribution is used as the mixing distribution
    in the variance-mean mixture representation of NIG. This function implements
    an efficient algorithm for generating IG random variates.
    
    Parameters
    ----------
    delta : float
        Location parameter of IG distribution. Must be positive.
        Related to the mean: E[X] = delta/gamma.
    gamma : float
        Scale parameter of IG distribution. Must be positive.
        Related to the shape of the distribution.
    M : int
        Number of rows in output array.
    N : int
        Number of columns in output array.
    
    Returns
    -------
    R : ndarray
        Array of shape (M, N) containing IG random variates.
    
    Notes
    -----
    The Inverse Gaussian distribution IG(delta, gamma) has PDF:
    
    .. math::
        f(x) = \\frac{\\gamma}{\\sqrt{2\\pi x^3}} 
               \\exp\\left(-\\frac{\\gamma^2(x-\\delta/\\gamma)^2}{2\\delta^2 x}\\right)
    
    **Generation algorithm:**
    Uses the algorithm from Michael, Schucany, and Haas (1976):
    1. Generate chi-squared(1) random variable V
    2. Compute two candidate values x1 and x2 from V
    3. Accept x1 with probability delta/(delta + gamma*x1), else accept x2
    
    This algorithm is exact and efficient, requiring only one chi-squared
    and one uniform random number per output.
    
    **Usage in NIG:**
    The NIG distribution can be represented as:
    X = mu + beta*Z + sqrt(Z)*Y
    where Z ~ IG(delta, gamma) and Y ~ N(0,1) are independent.
    
    Examples
    --------
    >>> import numpy as np
    >>> from stabpy.nig import invgrnd
    >>> # Generate IG random sample
    >>> sample = invgrnd(delta=1.0, gamma=2.0, M=1000, N=1)
    >>> print(f"Mean: {np.mean(sample):.3f} (expected: {1.0/2.0:.3f})")
    >>> print(f"Variance: {np.var(sample):.3f}")
    
    See Also
    --------
    nigrnd : Uses this function to generate NIG random numbers
    
    References
    ----------
    .. [1] J.R. Michael, W.R. Schucany, R.W. Haas (1976) "Generating random 
           variates using transformations with multiple roots", The American 
           Statistician, 30(2), 88-90.
    .. [2] S. Raible (2000) "Levy Processes in Finance - Theory, Numerics 
           and Empirical Facts", PhD Thesis, University of Freiburg.
    """
    # Step 1: Generate chi-squared(1) random variables
    # Chi-squared(1) is the square of a standard normal
    V = st.chi2.rvs(1, size=(M, N))
    
    # Step 2: Compute two candidate solutions from quadratic formula
    # These come from solving for x in the IG generating equation
    term = np.sqrt(4 * gamma * delta * V + V**2)
    x1 = delta / gamma + 1 / (2 * gamma**2) * (V + term)
    x2 = delta / gamma + 1 / (2 * gamma**2) * (V - term)
    
    # Step 3: Accept x1 with probability p1, otherwise accept x2
    Y = np.random.uniform(0, 1, (M, N))
    p1 = delta / (delta + gamma * x1)
    # p2 = 1 - p1  # (not needed, implicitly handled)
    
    # Create boolean mask for acceptance
    C = (Y < p1)
    
    # Select x1 where C is True, x2 where C is False
    R = C * x1 + (~C) * x2
    
    return R


def nigrnd(alpha=1, beta=0, mu=0, delta=1, m=1, n=1):
    """
    Generate random numbers from Normal Inverse Gaussian distribution.
    
    Uses the variance-mean mixture representation of NIG: generate Inverse
    Gaussian random variables as the mixing distribution, then generate
    conditionally normal variables given the mixing values.
    
    Parameters
    ----------
    alpha : float, optional
        Tail heaviness parameter. Must satisfy alpha > |beta|.
        Default is 1.
    beta : float, optional
        Asymmetry parameter. Must satisfy |beta| < alpha.
        Default is 0 (symmetric).
    mu : float, optional
        Location parameter. Default is 0.
    delta : float, optional
        Scale parameter. Must be positive. Default is 1.
    m : int, optional
        Number of rows in output array. Default is 1.
    n : int, optional
        Number of columns in output array. Default is 1.
    
    Returns
    -------
    r : ndarray
        Array of shape (m, n) containing NIG random variates.
    
    Raises
    ------
    ValueError
        If parameter constraints are violated:
        - alpha <= 0
        - delta <= 0
        - |beta| >= alpha
    
    Notes
    -----
    **Variance-mean mixture representation:**
    
    The NIG distribution can be represented as:
    
    .. math::
        X = \\mu + \\beta Z + \\sqrt{Z} Y
    
    where:
    - Z ~ IG(delta, gamma) with gamma = sqrt(alpha^2 - beta^2)
    - Y ~ N(0, 1) independent of Z
    
    This representation makes simulation straightforward and exact.
    
    **Statistical properties:**
    - Mean: mu + delta*beta/sqrt(alpha^2 - beta^2) * K_1/K_0
    - Variance: delta*alpha^2/(alpha^2 - beta^2)^(3/2) * K_1/K_0
    - Skewness: depends on beta (0 when beta=0)
    - Excess kurtosis: always positive (heavier tails than normal)
    
    **Parameter effects:**
    - Increasing alpha → lighter tails (approaches normal as alpha → ∞)
    - Increasing |beta| → more skewness
    - Increasing delta → more concentrated around mean
    
    Examples
    --------
    >>> import numpy as np
    >>> from stabpy.nig import nigrnd
    >>> # Generate symmetric NIG sample
    >>> np.random.seed(42)
    >>> x_sym = nigrnd(alpha=1.5, beta=0.0, mu=0.0, delta=1.0, m=1000, n=1)
    >>> print(f"Mean: {np.mean(x_sym):.3f}, Std: {np.std(x_sym):.3f}")
    >>> 
    >>> # Generate skewed NIG sample
    >>> x_skew = nigrnd(alpha=1.5, beta=0.8, mu=0.0, delta=1.0, m=1000, n=1)
    >>> print(f"Skewness: {np.mean(((x_skew - np.mean(x_skew))/np.std(x_skew))**3):.3f}")
    >>> 
    >>> # Generate matrix of NIG values
    >>> X = nigrnd(alpha=2.0, beta=0.5, mu=1.0, delta=1.5, m=10, n=5)
    >>> print(f"Shape: {X.shape}")
    
    See Also
    --------
    invgrnd : Generates the IG mixing distribution
    nigpdf : Probability density function
    nigest : Parameter estimation
    
    References
    ----------
    .. [1] K. Prause (1999) "The Generalized Hyperbolic Model"
    .. [2] O. Barndorff-Nielsen (1997) "Normal Inverse Gaussian Distributions"
    """
    # Validate parameter constraints
    if alpha <= 0:
        raise ValueError('ALPHA must be positive.')
    if delta <= 0:
        raise ValueError('DELTA must be positive.')
    if abs(beta) >= alpha:
        raise ValueError('BETA must satisfy |beta| < alpha.')
    
    # Compute gamma parameter for IG distribution
    gamma = np.sqrt(alpha**2 - beta**2)
    
    # Step 1: Generate Inverse Gaussian mixing variables
    # Z ~ IG(delta, gamma)
    x = invgrnd(delta, gamma, m, n)
    
    # Step 2: Generate standard normal variables
    # Y ~ N(0, 1)
    y = st.norm.rvs(size=(m, n))
    
    # Step 3: Construct NIG variates using variance-mean mixture
    # X = mu + beta*Z + sqrt(Z)*Y
    r = np.sqrt(x) * y + mu + beta * x
    
    return r


def nigtest(x, ilp):
    """
    Goodness-of-fit test for Normal Inverse Gaussian distribution.
    
    Performs multiple goodness-of-fit tests for the NIG distribution hypothesis.
    Parameters are estimated using maximum likelihood, and p-values are computed
    via Monte Carlo simulation.
    
    Parameters
    ----------
    x : array_like
        Sample data to test.
    ilp : int
        Number of Monte Carlo iterations for p-value estimation.
        Larger values give more accurate p-values but take longer.
        Recommended: at least 1000, preferably 2000+ for reliable results.
    
    Returns
    -------
    y : ndarray
        Array of length 14 containing:
        
        - y[0:4]: Estimated parameters [alpha, beta, delta, mu]
        - y[4:9]: Test statistics [D, V, W2, U2, A2] where:
            * D: Kolmogorov-Smirnov statistic
            * V: Kuiper statistic
            * W2: Cramér-von Mises statistic
            * U2: Watson statistic
            * A2: Anderson-Darling statistic
        - y[9:14]: P-values for each test statistic
    
    Notes
    -----
    **Test procedure:**
    
    1. Estimate parameters using MLE (nigest)
    2. Transform data to uniform [0,1] via estimated CDF
    3. Compute five test statistics measuring distance to uniform
    4. Generate ilp random samples from estimated NIG distribution
    5. Repeat steps 1-3 for each simulated sample
    6. P-value = proportion of simulated statistics exceeding observed
    
    **Interpretation:**
    
    - **p-value > 0.05**: Data is consistent with NIG distribution
    - **p-value < 0.05**: Significant evidence against NIG hypothesis
    - **p-value < 0.01**: Strong evidence against NIG
    
    Different tests have different sensitivities:
    - **KS (D)**: General purpose, sensitive throughout distribution
    - **Kuiper (V)**: More sensitive to tails than KS
    - **Cramér-von Mises (W²)**: Emphasizes body over tails
    - **Watson (U²)**: Location-invariant version of W²
    - **Anderson-Darling (A²)**: Most sensitive to tail deviations
    
    **Computational cost:**
    This test is computationally intensive because it requires:
    - MLE optimization for each of ilp+1 samples
    - Numerical integration of CDF for each sample
    
    For large samples (n > 500) or many iterations (ilp > 2000),
    expect significant computation time (minutes to hours).
    
    **When to use:**
    - Testing if financial returns follow NIG distribution
    - Model selection: comparing NIG to other heavy-tailed distributions
    - Validating simulations or theoretical models
    
    Examples
    --------
    >>> import numpy as np
    >>> from stabpy.nig import nigrnd, nigtest
    >>> # Test 1: Data from NIG distribution (should not reject)
    >>> np.random.seed(42)
    >>> x_nig = nigrnd(1.5, 0.5, 0.0, 1.0, 500, 1).flatten()
    >>> results = nigtest(x_nig, ilp=1000)
    >>> print(f"Estimated alpha: {results[0]:.3f}")
    >>> print(f"KS p-value: {results[9]:.3f}")  # Should be > 0.05
    >>> print(f"AD p-value: {results[13]:.3f}")
    >>> 
    >>> # Test 2: Data from normal distribution (should reject)
    >>> x_normal = np.random.randn(500)
    >>> results = nigtest(x_normal, ilp=1000)
    >>> print(f"KS p-value: {results[9]:.3f}")  # Should be < 0.05
    >>> 
    >>> # Test 3: Interpreting all tests
    >>> x = nigrnd(2.0, 0.8, 1.0, 1.5, 300, 1).flatten()
    >>> results = nigtest(x, ilp=2000)
    >>> test_names = ['KS', 'Kuiper', 'CvM', 'Watson', 'AD']
    >>> print(f"\\nEstimated parameters:")
    >>> print(f"alpha={results[0]:.3f}, beta={results[1]:.3f}, "
    ...       f"delta={results[2]:.3f}, mu={results[3]:.3f}")
    >>> print(f"\\nTest results:")
    >>> for i, name in enumerate(test_names):
    ...     stat = results[4+i]
    ...     pval = results[9+i]
    ...     print(f"{name:8s}: statistic={stat:.4f}, p-value={pval:.4f}")
    
    See Also
    --------
    nigest : Parameter estimation used in this test
    nigcdf : CDF computation used for transforming data
    nigtest : Similar test for normal distribution
    
    References
    ----------
    .. [1] R. Weron (2004) "Computationally intensive Value at Risk calculations"
    .. [2] K. Burnecki, J. Janczura, R. Weron (2011) "Building loss models"
    .. [3] M.A. Stephens (1974) "EDF Statistics for Goodness of Fit"
    """
    # Get sample size
    n = len(x)
    
    # Estimate parameters using Maximum Likelihood
    params = nigest(x)
    alphap, betap, deltap, mup = params
    
    # Initialize output array
    y = np.zeros(14)
    
    # Transform data to uniform [0,1] via estimated CDF
    z = np.sort(nigcdf(x, *params))
    
    # --- Compute test statistics ---
    
    # Kolmogorov-Smirnov: max vertical distance
    kp = np.max(np.arange(1, n + 1) / n - z)
    km = np.max(z - np.arange(n) / n)
    Dp = np.max([kp, km]) * (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n))
    
    # Kuiper: sum of max positive and negative distances
    Vp = (kp + km) * (np.sqrt(n) + 0.155 + 0.24 / np.sqrt(n))
    
    # Cramér-von Mises: integrated squared distance
    W2 = np.sum((z - np.arange(1, 2 * n, 2) / (2 * n)) ** 2) + 1 / (12 * n)
    
    # Watson: Cramér-von Mises with location correction
    U2 = W2 - n * (np.mean(z) - 0.5) ** 2
    
    # Apply finite-sample corrections
    W2p = (W2 - 0.4 / n + 0.6 / n ** 2) * (1 + 1 / n)
    U2p = (U2 - 0.1 / n + 0.1 / n ** 2) * (1 + 0.8 / n)
    
    # Anderson-Darling: weighted squared distance (emphasizes tails)
    A2p = -1 / n * np.sum(np.arange(1, 2 * n, 2) * 
                          (np.log(z) + np.log(1 - z[::-1]))) - n
    
    # --- Monte Carlo simulation for p-values ---
    
    # Generate ilp random samples from estimated NIG distribution
    proba = nigrnd(*params, n, ilp)
    
    # Storage for simulated test statistics
    probat = np.zeros((ilp, 5))
    
    for i in range(ilp):
        # Estimate parameters for this simulated sample
        params_i = nigest(proba[:, i])
        
        # Transform to uniform via estimated CDF
        z_i = np.sort(nigcdf(proba[:, i], *params_i))
        
        # Handle numerical edge cases (very rare)
        # Replace exact 0 or 1 values to avoid log(0) in AD statistic
        z_i[z_i == 0] = 3 * np.finfo(float).eps
        z_i[z_i == 1] = 1 - 3 * np.finfo(float).eps
        
        # Compute test statistics for simulated data
        kp_i = np.max(np.arange(1, n + 1) / n - z_i)
        km_i = np.max(z_i - np.arange(n) / n)
        
        probat[i, 0] = np.max([kp_i, km_i]) * (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n))
        probat[i, 1] = (kp_i + km_i) * (np.sqrt(n) + 0.155 + 0.24 / np.sqrt(n))
        
        W2i = np.sum((z_i - np.arange(1, 2 * n, 2) / (2 * n)) ** 2) + 1 / (12 * n)
        U2i = W2i - n * (np.mean(z_i) - 0.5) ** 2
        
        probat[i, 2] = (W2i - 0.4 / n + 0.6 / n ** 2) * (1 + 1 / n)
        probat[i, 3] = (U2i - 0.1 / n + 0.1 / n ** 2) * (1 + 0.8 / n)
        probat[i, 4] = -1 / n * np.sum(np.arange(1, 2 * n, 2) * 
                                        (np.log(z_i) + np.log(1 - z_i[::-1]))) - n

    
    # Store estimated parameters
    y[:4] = [alphap, betap, deltap, mup]
    
    # Store observed test statistics
    pt = [Dp, Vp, W2p, U2p, A2p]
    y[4:9] = pt
    
    # Compute p-values: proportion of simulated statistics exceeding observed
    pom = np.zeros(5)
    for i in range(5):
        pom[i] = np.sum(probat[:, i] > pt[i]) / ilp
    
    y[9:14] = pom
    
    return y
