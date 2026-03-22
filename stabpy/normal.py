"""
Normal Distribution Testing
============================

This module provides goodness-of-fit tests for the normal (Gaussian) distribution.
Tests include Kolmogorov-Smirnov, Kuiper, Cramér-von Mises, Watson, and 
Anderson-Darling statistics with Monte Carlo p-value estimation.

References
----------
.. [1] Sz. Borak, A. Misiorek, R. Weron (2011) "Models for heavy-tailed asset 
       returns", in "Statistical Tools for Finance and Insurance, 2nd Edition", 
       Springer, Berlin, 21-56.
.. [2] K. Burnecki, J. Janczura, R. Weron (2011) "Building loss models", in 
       "Statistical Tools for Finance and Insurance, 2nd Edition", Springer, 
       Berlin, 293-328.
.. [3] M.A. Stephens (1974) "EDF Statistics for Goodness of Fit and Some 
       Comparisons", Journal of the American Statistical Association, 69(347), 
       730-737.
"""

import numpy as np
import scipy.stats as st


def normtest(x, ilp):
    """
    Goodness-of-fit test for normal (Gaussian) distribution.
    
    Performs multiple goodness-of-fit tests for the normal distribution hypothesis.
    Parameters are estimated using maximum likelihood estimation (MLE), and p-values
    are computed via Monte Carlo simulation. This function tests whether the data
    could reasonably come from a normal distribution.
    
    Parameters
    ----------
    x : array_like
        Sample data to test for normality. Should be 1-dimensional.
    ilp : int
        Number of Monte Carlo iterations for p-value estimation.
        Larger values give more accurate p-values but take longer.
        Recommended: at least 1000 for reliable results.
    
    Returns
    -------
    y : ndarray
        Array of shape (1, 12) containing test results:
        
        - y[0, 0:2]: Estimated parameters [mu, sigma]
            * mu: sample mean (location parameter)
            * sigma: sample standard deviation (scale parameter)
        
        - y[0, 2:7]: Test statistics [D, V, W2, U2, A2]
            * D: Kolmogorov-Smirnov statistic (maximum vertical distance)
            * V: Kuiper statistic (sum of max positive and negative distances)
            * W2: Cramér-von Mises statistic (integrated squared distance)
            * U2: Watson statistic (Cramér-von Mises with location correction)
            * A2: Anderson-Darling statistic (weighted squared distance)
        
        - y[0, 7:12]: P-values for each test statistic
            P-value = proportion of simulated statistics exceeding observed value
    
    Notes
    -----
    The testing procedure consists of:
    
    1. **Parameter Estimation**: Uses scipy's MLE to estimate mu and sigma
    2. **Empirical CDF**: Computes the empirical cumulative distribution function
    3. **Test Statistics**: Calculates five different statistics measuring
       the distance between empirical and theoretical CDFs
    4. **Monte Carlo Simulation**: Generates `ilp` random samples from the
       estimated normal distribution
    5. **P-value Calculation**: For each test, computes the proportion of
       simulated statistics that exceed the observed statistic
    
    **Interpretation of test statistics:**
    
    - **Kolmogorov-Smirnov (D)**: Measures the maximum absolute difference
      between empirical and theoretical CDFs. Sensitive to differences anywhere
      in the distribution.
    
    - **Kuiper (V)**: Similar to KS but more sensitive to differences in tails
      by considering both positive and negative deviations separately.
    
    - **Cramér-von Mises (W²)**: Integrates squared deviations across the
      entire distribution. More sensitive to differences in the body than KS.
    
    - **Watson (U²)**: Like Cramér-von Mises but invariant to location shifts.
      Good for detecting differences in shape.
    
    - **Anderson-Darling (A²)**: Weighted version of Cramér-von Mises that
      gives more weight to tail deviations. Most powerful for detecting
      heavy tails or tail asymmetry.
    
    **Interpretation of p-values:**
    
    - p-value > 0.05: Data is consistent with normal distribution (fail to reject)
    - p-value < 0.05: Data significantly deviates from normal (reject normality)
    - p-value < 0.01: Strong evidence against normality
    
    Note that different tests may give different results. Multiple significant
    p-values provide stronger evidence against normality.
    
    **Finite-sample corrections:**
    
    The test statistics include finite-sample corrections to improve accuracy
    for small sample sizes. These corrections are from Stephens (1974).
    
    Examples
    --------
    >>> import numpy as np
    >>> from stabpy.normal import normtest
    >>> 
    >>> # Test 1: Data from normal distribution (should not reject)
    >>> np.random.seed(42)
    >>> x_normal = np.random.randn(100)
    >>> results = normtest(x_normal, ilp=1000)
    >>> print(f"Mean: {results[0, 0]:.3f}, Std: {results[0, 1]:.3f}")
    >>> print(f"KS p-value: {results[0, 7]:.3f}")  # Should be > 0.05
    >>> 
    >>> # Test 2: Data from heavy-tailed distribution (should reject)
    >>> from scipy.stats import t
    >>> x_heavy = t.rvs(df=3, size=100)  # Student's t with 3 df
    >>> results = normtest(x_heavy, ilp=1000)
    >>> print(f"AD p-value: {results[0, 11]:.3f}")  # Should be < 0.05
    >>> 
    >>> # Test 3: Interpreting all tests
    >>> x = np.random.randn(200)
    >>> results = normtest(x, ilp=2000)
    >>> test_names = ['KS', 'Kuiper', 'CvM', 'Watson', 'AD']
    >>> for i, name in enumerate(test_names):
    ...     stat = results[0, 2+i]
    ...     pval = results[0, 7+i]
    ...     print(f"{name}: statistic={stat:.4f}, p-value={pval:.4f}")
    
    See Also
    --------
    scipy.stats.normaltest : Alternative normality test (D'Agostino-Pearson)
    scipy.stats.shapiro : Shapiro-Wilk normality test
    scipy.stats.kstest : Kolmogorov-Smirnov test with known parameters
    
    References
    ----------
    .. [1] Sz. Borak, A. Misiorek, R. Weron (2011) "Models for heavy-tailed asset 
           returns", in "Statistical Tools for Finance and Insurance, 2nd Edition", 
           Springer, Berlin, 21-56.
    .. [2] K. Burnecki, J. Janczura, R. Weron (2011) "Building loss models", in 
           "Statistical Tools for Finance and Insurance, 2nd Edition", Springer, 
           Berlin, 293-328.
    .. [3] M.A. Stephens (1974) "EDF Statistics for Goodness of Fit and Some 
           Comparisons", Journal of the American Statistical Association, 69(347), 
           730-737.
    .. [4] R.B. D'Agostino and M.A. Stephens (1986) "Goodness-of-Fit Techniques",
           Marcel Dekker, New York.
    """
    # Get sample size
    n = len(x)
    
    # Estimate parameters using Maximum Likelihood Estimation
    # For normal distribution: MLE gives sample mean and sample std
    mu, sigma = st.norm.fit(x)
    mup = mu      # Store for output
    sigmap = sigma
    
    # Initialize output array: [mu, sigma, 5 statistics, 5 p-values]
    y = np.zeros((1, 12))
    
    # Compute empirical CDF by transforming data to uniform [0,1]
    # under the estimated normal distribution
    z = st.norm.cdf(x, mu, sigma)
    z.sort()
    
    # --- Kolmogorov-Smirnov statistic ---
    # Measures maximum vertical distance between empirical and theoretical CDF
    # D = max(|F_n(x) - F(x)|)
    kp = np.max((np.arange(1, n + 1).reshape(-1, 1))/n - z.reshape(-1, 1))  # Max positive deviation
    km = np.max(z.reshape(-1, 1) - np.arange(0, n).reshape(-1, 1)/n)        # Max negative deviation
    # Apply finite-sample correction
    Dp = np.max([kp, km]) * (np.sqrt(n) + 0.12 + 0.11/np.sqrt(n))
    
    # --- Kuiper statistic ---
    # Sum of maximum positive and negative deviations
    # V = max(F_n - F) + max(F - F_n)
    # More sensitive to tail differences than KS
    Vp = (kp + km) * (np.sqrt(n) + 0.155 + 0.24/np.sqrt(n))
    
    # --- Cramér-von Mises statistic ---
    # Integrates squared deviations: W² = ∫[F_n(x) - F(x)]² dF(x)
    # Approximated using sum over sample points
    W2 = np.sum((z.reshape(-1, 1) - (np.arange(1, 2*n, 2).reshape(-1, 1) / (2*n)))**2) + 1/(12*n)
    
    # --- Watson statistic ---
    # Modified Cramér-von Mises that is invariant to location shifts
    # U² = W² - n(z̄ - 0.5)²
    U2 = W2 - n * (np.mean(z) - 1/2)**2
    
    # Apply finite-sample corrections to W² and U²
    W2p = (W2 - 0.4/n + 0.6/n**2) * (1 + 1/n)
    U2p = (U2 - 0.1/n + 0.1/n**2) * (1 + 0.8/n)
    
    # --- Anderson-Darling statistic ---
    # Weighted Cramér-von Mises giving more weight to tails
    # A² = -n - (1/n) Σ[(2i-1)(ln F(X_i) + ln(1-F(X_{n+1-i})))]
    # Most sensitive to deviations in the tails
    A2p = -1/n * np.dot(np.arange(1, 2*n, 2), 
                         (np.log(z) + np.log(1 - z[n-1::-1]))) - n
    
    # --- Monte Carlo simulation for p-value estimation ---
    # Generate ilp random samples from the estimated normal distribution
    proba = sigma * np.random.randn(n, ilp) + mu
    probat = np.zeros((ilp, 5))  # Store statistics for each simulation
    
    for i in range(ilp):
        # Estimate parameters for this simulated sample
        mu1, sigma1 = st.norm.fit(proba[:, i])
        z1 = st.norm.cdf(proba[:, i], mu1, sigma1)
        z1.sort()
        
        # Compute KS statistic for simulated data
        kp1 = np.max((np.arange(1, n + 1).reshape(-1, 1))/n - z1.reshape(-1, 1))
        km1 = np.max(z1.reshape(-1, 1) - np.arange(0, n).reshape(-1, 1)/n)
        probat[i, 0] = np.max([kp1, km1]) * (np.sqrt(n) + 0.12 + 0.11/np.sqrt(n))
        
        # Compute Kuiper statistic for simulated data
        probat[i, 1] = (kp1 + km1) * (np.sqrt(n) + 0.155 + 0.24/np.sqrt(n))
        
        # Compute Cramér-von Mises and Watson statistics for simulated data
        W2 = np.sum((z1.reshape(-1, 1) - (np.arange(1, 2*n, 2).reshape(-1, 1) / (2*n)))**2) + 1/(12*n)
        U2 = W2 - n * (np.mean(z1) - 1/2)**2
        probat[i, 2] = (W2 - 0.4/n + 0.6/n**2) * (1 + 1/n)
        probat[i, 3] = (U2 - 0.1/n + 0.1/n**2) * (1 + 0.8/n)
        
        # Compute Anderson-Darling statistic for simulated data
        probat[i, 4] = -1/n * np.dot(np.arange(1, 2*n, 2), 
                                      (np.log(z1) + np.log(1 - z1[n-1::-1]))) - n
    
    # --- Compile results ---
    # Store estimated parameters
    y[0, 0:2] = [mup, sigmap]
    
    # Store observed test statistics
    pt = [Dp, Vp, W2p, U2p, A2p]
    y[0, 2:7] = pt
    
    # Compute p-values: proportion of simulated statistics exceeding observed
    # p-value = P(statistic from null distribution > observed statistic)
    # Small p-values indicate the data is unlikely to come from normal distribution
    pom = np.zeros(5)
    for i in range(5):
        pom[i] = np.sum(probat[:, i] > pt[i]) / ilp
    y[0, 7:12] = pom
    
    return y
