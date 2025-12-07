import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --- 1. Simulation Settings ---
np.random.seed(42)
N_sim = 100000       
# Simple setup: theta ~ N(0,1), x1 ~ N(theta, 1), x2 ~ N(theta, 1)
# Full Data: (x1, x2). Lossy Data: x1.
tau_prior = 1.0 
tau_obs = 1.0

# Generate Data
thetas = np.random.normal(0, 1, N_sim)
x1 = np.random.normal(thetas, 1)
x2 = np.random.normal(thetas, 1)

# --- 2. Analytic Posteriors ---

# A. Ground Truth (Full Information)
# Precision = 1 (prior) + 1 (x1) + 1 (x2) = 3
# Mean = (0 + x1 + x2) / 3
sigma_full = np.sqrt(1/3)
mu_full = (x1 + x2) / 3

# B. Lossy Summary (Only x1)
# Precision = 1 (prior) + 1 (x1) = 2
# Mean = x1 / 2
sigma_lossy = np.sqrt(1/2)
mu_lossy = x1 / 2

# C. Overdispersed (Miscalibrated)
# Uses Full Mean (correct information) but Lossy Variance (over-cautious)
# This mimics a model that extracts features well but inflates uncertainty.
mu_over = mu_full
sigma_over = sigma_lossy # Wider than sigma_full

# --- 3. Compute Ranks for SBC ---

# Rank of True Theta in Lossy Posterior
# Valid because theta, x1 are drawn from the joint.
ranks_lossy = stats.norm.cdf(thetas, loc=mu_lossy, scale=sigma_lossy)

# Rank of True Theta in Overdispersed Posterior
# Invalid model: assumes variance is 1/2 when true conditional variance is 1/3
ranks_over = stats.norm.cdf(thetas, loc=mu_over, scale=sigma_over)

# --- 4. Plotting Setup ---
c_truth = '#333333'      # Grey
c_lossy = '#DDAA33'      # Mustard (Type B: Lossy)
c_over = '#BB5566'       # Rose (Type C: Miscalibrated)

fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.2))

plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'legend.fontsize': 7,
    'font.family': 'sans-serif' 
})

# === LEFT PANEL: Ridge Plot (Joyplot) ===
ax0 = axes[0]
t_grid = np.linspace(-3, 3, 500)

# Select representative samples for visualization
# We sort by theta to make the ridge plot flow nicely
sort_idx = np.argsort(thetas)
# Pick 6 evenly spaced indices from the bulk
viz_indices = sort_idx[np.linspace(10000, 90000, 6, dtype=int)]

y_offsets = np.arange(len(viz_indices)) * 0.8

for i, idx in enumerate(viz_indices):
    base_y = y_offsets[i]
    
    # 1. Ground Truth (Full Info)
    pdf_true = stats.norm.pdf(t_grid, loc=mu_full[idx], scale=sigma_full)
    ax0.plot(t_grid, pdf_true + base_y, color=c_truth, ls='--', lw=1.2, zorder=10)
    
    # 2. Overdispersed (Centered on Truth, but Wide)
    pdf_over = stats.norm.pdf(t_grid, loc=mu_over[idx], scale=sigma_over)
    ax0.plot(t_grid, pdf_over + base_y, color=c_over, lw=2.0, zorder=5)
    
    # 3. Lossy (Shifted center, Wide)
    pdf_lossy = stats.norm.pdf(t_grid, loc=mu_lossy[idx], scale=sigma_lossy)
    ax0.plot(t_grid, pdf_lossy + base_y, color=c_lossy, lw=2.0, zorder=8)
    # Fill Lossy for visibility
    ax0.fill_between(t_grid, base_y, pdf_lossy + base_y, color=c_lossy, alpha=0.2)

# Dummy Legend
ax0.plot([], [], color=c_truth, ls='--', label='Ground Truth $p(\\theta|x_{1}, x_{2})$')
ax0.plot([], [], color=c_over, lw=2, label='Overdispersed (Miscalibrated)')
ax0.plot([], [], color=c_lossy, lw=2, label='Lossy Summary $p(\\theta|x_{1})$')

ax0.set_xlabel(r'$\theta$')
ax0.set_ylabel('Posterior Density (Offset)')
ax0.set_yticks([])
ax0.set_xlim(-3, 3)
ax0.set_ylim(-0.2, 6.3)
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.spines['left'].set_visible(False)
ax0.legend(loc='upper left', frameon=False)
ax0.text(-0.05, 1.0, '(a)', transform=ax0.transAxes, fontweight='bold', fontsize=10, va='top', ha='right')
#ax0.set_title('Posterior Width vs Information', fontsize=9)

# === RIGHT PANEL: SBC Rank Histograms ===
ax1 = axes[1]
n_bins = 20
bin_edges = np.linspace(0, 1, n_bins + 1)
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

# 1. Lossy (Mustard) -> Uniform
# Because it is a valid posterior given x1. The width correctly reflects the lack of x2.
hist_lossy, _ = np.histogram(ranks_lossy, bins=bin_edges, density=True)
ax1.step(bin_centers, hist_lossy, where='mid', color=c_lossy, lw=2.5, label='Lossy Summary')

# 2. Overdispersed (Rose) -> Inverted U
# Because it is wider than the ground truth (Full), but centered on it.
hist_over, _ = np.histogram(ranks_over, bins=bin_edges, density=True)
ax1.step(bin_centers, hist_over, where='mid', color=c_over, lw=2.5, label='Overdispersed')

# 3. Reference
ax1.axhline(1.0, color='gray', ls='--', lw=1.0, alpha=0.5, label='Ideal Uniformity')

ax1.set_xlabel(r'Rank Statistic $U(\theta^*)$')
ax1.set_ylabel('Normalized Density')
ax1.set_ylim(0.0, 2.0)
ax1.set_xlim(0, 1)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax1.legend(loc='upper center', frameon=False, fontsize=8)
ax1.text(-0.18, 1.0, '(b)', transform=ax1.transAxes, fontweight='bold', fontsize=10, va='top', ha='right')
#ax1.set_title('SBC Rank Diagnostics', fontsize=9)

# Annotation
#ax1.text(0.5, 0.4, 'Lossy is calibrated!\n(Uniform Rank)', color=c_lossy, 
#         ha='center', va='center', fontsize=8, fontweight='bold', alpha=0.9)
#ax1.text(0.5, 1.6, 'Overdispersed\nis underconfident', color=c_over, 
#         ha='center', va='center', fontsize=8, fontweight='bold', alpha=0.9)

plt.tight_layout()
plt.savefig('figures/fig_sbc_lossy.pdf', format='pdf', bbox_inches='tight')
plt.show()