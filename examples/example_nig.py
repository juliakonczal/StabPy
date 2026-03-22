"""
Example: Normal Inverse Gaussian Distribution
==============================================

This example demonstrates:
1. Generating NIG random samples
2. Estimating parameters via MLE
3. Computing PDF and CDF
4. Testing goodness of fit
"""

import numpy as np
import matplotlib.pyplot as plt
from stabpy import nigrnd, nigest, nigpdf, nigcdf, nigtest

# Set random seed
np.random.seed(42)

print("=" * 60)
print("Normal Inverse Gaussian Distribution Example")
print("=" * 60)

# 1. Generate NIG random sample
print("\n1. Generating NIG random sample...")
print("   Parameters: α=2.0, β=0.8, δ=1.5, μ=1.0")

alpha_true = 2.0
beta_true = 0.8
delta_true = 1.5
mu_true = 1.0

x = nigrnd(alpha_true, beta_true, mu_true, delta_true, 500, 1).flatten()

print(f"   Generated {len(x)} samples")
print(f"   Sample mean: {np.mean(x):.3f}")
print(f"   Sample std: {np.std(x):.3f}")

# 2. Estimate parameters using MLE
print("\n2. Parameter estimation (MLE)...")
params = nigest(x)
alpha_est, beta_est, delta_est, mu_est = params

print(f"   Estimated: α={alpha_est:.3f}, β={beta_est:.3f}, δ={delta_est:.3f}, μ={mu_est:.3f}")
print(f"   True:      α={alpha_true:.3f}, β={beta_true:.3f}, δ={delta_true:.3f}, μ={mu_true:.3f}")

# 3. Compute PDF and CDF
print("\n3. Computing PDF and CDF...")
x_grid = np.linspace(np.min(x) - 2, np.max(x) + 2, 200)
pdf_vals = nigpdf(x_grid, alpha_est, beta_est, delta_est, mu_est)
cdf_vals = nigcdf(x_grid, alpha_est, beta_est, delta_est, mu_est)

print(f"   Computed PDF and CDF on grid of {len(x_grid)} points")

# 4. Goodness of fit test
print("\n4. Goodness of fit test (this may take a few minutes)...")
results = nigtest(x, ilp=200)

test_names = ['Kolmogorov-Smirnov', 'Kuiper', 'Cramér-von Mises', 'Watson', 'Anderson-Darling']
print("\n   Test results:")
for i, name in enumerate(test_names):
    stat = results[4 + i]
    pval = results[9 + i]
    print(f"   {name:20s}: statistic={stat:.4f}, p-value={pval:.4f}")

# 5. Visualize results
print("\n5. Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Histogram with estimated PDF
ax1 = axes[0, 0]
ax1.hist(x, bins=40, density=True, alpha=0.7, edgecolor='black', label='Data')
ax1.plot(x_grid, pdf_vals, 'r-', linewidth=2, label='Estimated PDF')
ax1.set_xlabel('Value')
ax1.set_ylabel('Density')
ax1.set_title('NIG Distribution: Histogram and Estimated PDF')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Empirical vs Theoretical CDF
ax2 = axes[0, 1]
x_sorted = np.sort(x)
empirical_cdf = np.arange(1, len(x)+1) / len(x)
theoretical_cdf = nigcdf(x_sorted, alpha_est, beta_est, delta_est, mu_est)
ax2.plot(x_sorted, empirical_cdf, 'b-', linewidth=1.5, label='Empirical CDF', alpha=0.7)
ax2.plot(x_sorted, theoretical_cdf, 'r--', linewidth=2, label='Theoretical CDF')
ax2.set_xlabel('Value')
ax2.set_ylabel('CDF')
ax2.set_title('Cumulative Distribution Functions')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Q-Q plot
ax3 = axes[1, 0]
ax3.scatter(theoretical_cdf, empirical_cdf, alpha=0.5, s=20)
ax3.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect fit')
ax3.set_xlabel('Theoretical Quantiles')
ax3.set_ylabel('Empirical Quantiles')
ax3.set_title('Q-Q Plot')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Parameter comparison
ax4 = axes[1, 1]
params_true = [alpha_true, beta_true, delta_true, mu_true]
params_est = [alpha_est, beta_est, delta_est, mu_est]

x_pos = np.arange(4)
width = 0.35

ax4.bar(x_pos - width/2, params_true, width, label='True', alpha=0.8)
ax4.bar(x_pos + width/2, params_est, width, label='Estimated', alpha=0.8)

ax4.set_xlabel('Parameter')
ax4.set_ylabel('Value')
ax4.set_title('Parameter Estimation Results')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(['α', 'β', 'δ', 'μ'])
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('nig_example.png', dpi=150, bbox_inches='tight')
print("   Saved plot to 'nig_example.png'")

# 6. Demonstrate skewness effect
print("\n6. Comparing symmetric vs skewed NIG...")

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Symmetric (beta=0)
x_sym = nigrnd(2.0, 0.0, 0.0, 1.0, 1000, 1).flatten()
ax1.hist(x_sym, bins=40, density=True, alpha=0.7, edgecolor='black')
ax1.set_title('Symmetric NIG (β=0)')
ax1.set_xlabel('Value')
ax1.set_ylabel('Density')
ax1.grid(True, alpha=0.3)

# Skewed (beta=0.8)
x_skew = nigrnd(2.0, 0.8, 0.0, 1.0, 1000, 1).flatten()
ax2.hist(x_skew, bins=40, density=True, alpha=0.7, edgecolor='black', color='orange')
ax2.set_title('Right-skewed NIG (β=0.8)')
ax2.set_xlabel('Value')
ax2.set_ylabel('Density')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('nig_skewness.png', dpi=150, bbox_inches='tight')
print("   Saved skewness comparison to 'nig_skewness.png'")

print("\n" + "=" * 60)
print("Example completed!")
print("=" * 60)
