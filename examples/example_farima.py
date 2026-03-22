"""
Example: FARIMA Processes and Fractional Brownian Motion
=========================================================

This example demonstrates:
1. Generating fractional Brownian motion
2. Simulating FARIMA processes
3. Estimating long-memory parameter
4. Visualizing long-range dependence
"""

import numpy as np
import matplotlib.pyplot as plt
from stabpy import usg, fftfarima, fftFarimaEst, fracdiff

# Set random seed
np.random.seed(42)

print("=" * 60)
print("FARIMA and Fractional Processes Example")
print("=" * 60)

# 1. Generate fractional Gaussian noise and fBm
print("\n1. Generating fractional Brownian motion...")

H_values = [0.3, 0.5, 0.7, 0.9]
fgn_samples = {}
fbm_samples = {}

for H in H_values:
    fgn = usg(H_=H, N_=10)  # 2^10 = 1024 samples
    fbm = np.cumsum(fgn)
    fgn_samples[H] = fgn
    fbm_samples[H] = fbm
    print(f"   H={H}: Generated fGn and fBm")

# 2. Generate FARIMA process
print("\n2. Generating FARIMA(1, 0.4, 1) process...")

d_true = 0.4
phi_true = 0.5
theta_true = 0.3

Y = fftfarima(
    alpha=2.0,  # Gaussian innovations
    d=d_true,
    n=1,
    Ph=[phi_true],
    Th=[theta_true],
    M=100,
    N=1000
)
x_farima = Y[0, :]

print(f"   Generated {len(x_farima)} observations")
print(f"   Sample mean: {np.mean(x_farima):.3f}")
print(f"   Sample std: {np.std(x_farima):.3f}")

# 3. Estimate FARIMA parameters
print("\n3. Estimating FARIMA parameters...")
params_est = fftFarimaEst(x_farima, p=1, q=1)
d_est, phi_est, theta_est = params_est

print(f"   Estimated: d={d_est:.3f}, φ={phi_est:.3f}, θ={theta_est:.3f}")
print(f"   True:      d={d_true:.3f}, φ={phi_true:.3f}, θ={theta_true:.3f}")

# 4. Compute ACF for different d values
print("\n4. Computing autocorrelation functions...")

def compute_acf(x, max_lag=50):
    """Compute sample autocorrelation function."""
    acf = np.correlate(x - np.mean(x), x - np.mean(x), mode='full')
    acf = acf[len(acf)//2:]
    acf = acf / acf[0]
    return acf[:max_lag]

acf_farima = compute_acf(x_farima)
print(f"   Computed ACF for {len(acf_farima)} lags")

# 5. Visualize results
print("\n5. Creating visualizations...")

fig = plt.figure(figsize=(14, 10))

# Plot 1: Fractional Gaussian Noise
ax1 = plt.subplot(3, 2, 1)
for H in H_values:
    ax1.plot(fgn_samples[H][:200], alpha=0.7, label=f'H={H}')
ax1.set_title('Fractional Gaussian Noise')
ax1.set_xlabel('Time')
ax1.set_ylabel('Value')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Fractional Brownian Motion
ax2 = plt.subplot(3, 2, 2)
for H in H_values:
    ax2.plot(fbm_samples[H][:200], alpha=0.7, label=f'H={H}')
ax2.set_title('Fractional Brownian Motion')
ax2.set_xlabel('Time')
ax2.set_ylabel('Cumulative Sum')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: FARIMA process
ax3 = plt.subplot(3, 2, 3)
ax3.plot(x_farima[:500], linewidth=0.8)
ax3.set_title(f'FARIMA(1, {d_true}, 1) Process')
ax3.set_xlabel('Time')
ax3.set_ylabel('Value')
ax3.grid(True, alpha=0.3)

# Plot 4: ACF of FARIMA
ax4 = plt.subplot(3, 2, 4)
ax4.bar(range(len(acf_farima)), acf_farima, width=1.0, alpha=0.7, edgecolor='black')
ax4.axhline(y=0, color='r', linestyle='--', linewidth=1)
ax4.set_title('Autocorrelation Function')
ax4.set_xlabel('Lag')
ax4.set_ylabel('ACF')
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: Comparison of H values (variance of increments)
ax5 = plt.subplot(3, 2, 5)
scales = 2**np.arange(1, 8)
variances = {H: [] for H in H_values}

for H in H_values:
    fbm = fbm_samples[H]
    for scale in scales:
        increments = fbm[scale:] - fbm[:-scale]
        variances[H].append(np.var(increments))
    ax5.loglog(scales, variances[H], 'o-', label=f'H={H}', alpha=0.7)

ax5.set_title('Scaling of Variance (fBm)')
ax5.set_xlabel('Scale')
ax5.set_ylabel('Variance of Increments')
ax5.legend()
ax5.grid(True, alpha=0.3, which='both')

# Plot 6: Fractional differencing coefficients
ax6 = plt.subplot(3, 2, 6)
d_values = [0.2, 0.4, 0.6, 0.8]
for d in d_values:
    coefs = fracdiff(d, N=50)
    ax6.plot(np.abs(coefs), 'o-', alpha=0.7, label=f'd={d}', markersize=4)

ax6.set_title('Fractional Differencing Coefficients')
ax6.set_xlabel('Lag')
ax6.set_ylabel('|Coefficient|')
ax6.set_yscale('log')
ax6.legend()
ax6.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('farima_example.png', dpi=150, bbox_inches='tight')
print("   Saved plot to 'farima_example.png'")

# 6. Demonstrate heavy-tailed FARIMA
print("\n6. Comparing Gaussian vs heavy-tailed FARIMA...")

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Gaussian FARIMA
Y_gauss = fftfarima(alpha=2.0, d=0.3, n=1, Ph=[0.5], Th=None, M=50, N=1000)
ax1.plot(Y_gauss[0, :], linewidth=0.8)
ax1.set_title('Gaussian FARIMA (α=2.0)')
ax1.set_xlabel('Time')
ax1.set_ylabel('Value')
ax1.grid(True, alpha=0.3)

# Heavy-tailed FARIMA
Y_stable = fftfarima(alpha=1.5, d=0.3, n=1, Ph=[0.5], Th=None, M=50, N=1000)
ax2.plot(Y_stable[0, :], linewidth=0.8, color='orange')
ax2.set_title('Heavy-tailed FARIMA (α=1.5)')
ax2.set_xlabel('Time')
ax2.set_ylabel('Value')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('farima_innovations.png', dpi=150, bbox_inches='tight')
print("   Saved innovations comparison to 'farima_innovations.png'")

print("\n" + "=" * 60)
print("Example completed!")
print("=" * 60)
print("\nKey insights:")
print("- H > 0.5: Persistent (positive autocorrelation)")
print("- H = 0.5: Independent increments (Brownian motion)")
print("- H < 0.5: Anti-persistent (negative autocorrelation)")
print("- FARIMA captures long-range dependence via parameter d")
