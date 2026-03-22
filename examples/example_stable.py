"""
Example: Working with Stable Distributions
===========================================

This example demonstrates:
1. Generating stable random samples
2. Estimating parameters
3. Testing goodness of fit
4. Visualizing results
"""

import numpy as np
import matplotlib.pyplot as plt
from stabpy import stabrnd, stabcull, stabreg, stabcdf, stabtest

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("Stable Distributions Example")
print("=" * 60)

# 1. Generate stable random sample
print("\n1. Generating stable random sample...")
print("   Parameters: α=1.5, β=0.5, σ=2.0, μ=1.0")

# Generate 1000 samples
alpha_true = 1.5
beta_true = 0.5
sigma_true = 2.0
mu_true = 1.0

x = sigma_true * stabrnd(alpha_true, beta_true, 1000, 1).flatten() + mu_true

print(f"   Generated {len(x)} samples")
print(f"   Sample mean: {np.mean(x):.3f}")
print(f"   Sample median: {np.median(x):.3f}")

# 2. Estimate parameters using quantile method (fast)
print("\n2. Parameter estimation (quantile method)...")
alpha_q, sigma_q, beta_q, mu_q = stabcull(x)

print(f"   Estimated: α={alpha_q:.3f}, σ={sigma_q:.3f}, β={beta_q:.3f}, μ={mu_q:.3f}")
print(f"   True:      α={alpha_true:.3f}, σ={sigma_true:.3f}, β={beta_true:.3f}, μ={mu_true:.3f}")

# 3. Estimate parameters using regression method (more accurate)
print("\n3. Parameter estimation (regression method)...")
alpha_r, sigma_r, beta_r, mu_r = stabreg(x)

print(f"   Estimated: α={alpha_r:.3f}, σ={sigma_r:.3f}, β={beta_r:.3f}, μ={mu_r:.3f}")
print(f"   True:      α={alpha_true:.3f}, σ={sigma_true:.3f}, β={beta_true:.3f}, μ={mu_true:.3f}")

# 4. Goodness of fit test
print("\n4. Goodness of fit test (this may take a minute)...")
results = stabtest(x, ilp=500)

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
ax1.hist(x, bins=50, density=True, alpha=0.7, edgecolor='black', label='Data')
ax1.set_xlabel('Value')
ax1.set_ylabel('Density')
ax1.set_title('Histogram of Stable Random Sample')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Q-Q plot
ax2 = axes[0, 1]
x_sorted = np.sort(x)
theoretical_quantiles = stabcdf(x_sorted, alpha_r, sigma_r, beta_r, mu_r)
empirical_quantiles = np.arange(1, len(x)+1) / (len(x)+1)
ax2.scatter(theoretical_quantiles, empirical_quantiles, alpha=0.5, s=10)
ax2.plot([0, 1], [0, 1], 'r--', label='Perfect fit')
ax2.set_xlabel('Theoretical Quantiles')
ax2.set_ylabel('Empirical Quantiles')
ax2.set_title('Q-Q Plot')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Time series plot
ax3 = axes[1, 0]
ax3.plot(x[:200], linewidth=0.8)
ax3.set_xlabel('Index')
ax3.set_ylabel('Value')
ax3.set_title('First 200 Samples')
ax3.grid(True, alpha=0.3)

# Comparison of estimation methods
ax4 = axes[1, 1]
params_true = [alpha_true, sigma_true, beta_true, mu_true]
params_quant = [alpha_q, sigma_q, beta_q, mu_q]
params_reg = [alpha_r, sigma_r, beta_r, mu_r]

x_pos = np.arange(4)
width = 0.25

ax4.bar(x_pos - width, params_true, width, label='True', alpha=0.8)
ax4.bar(x_pos, params_quant, width, label='Quantile', alpha=0.8)
ax4.bar(x_pos + width, params_reg, width, label='Regression', alpha=0.8)

ax4.set_xlabel('Parameter')
ax4.set_ylabel('Value')
ax4.set_title('Parameter Estimation Comparison')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(['α', 'σ', 'β', 'μ'])
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('stable_example.png', dpi=150, bbox_inches='tight')
print("   Saved plot to 'stable_example.png'")

print("\n" + "=" * 60)
print("Example completed!")
print("=" * 60)
