import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --- 1. Simulation Settings ---
np.random.seed(42)
N_sim = 100000       
sigma_true = 1.0     
bias_mag = 0.2       # Set to 0.0 to verify uniformity

# Data Generation (Wide Prior to eliminate boundary effects)
# We simulate a "Flat Prior" regime by making the bounds very far away
thetas = np.random.uniform(-20, 20, N_sim)
xs = np.random.normal(thetas, sigma_true)

# FILTER: Only analyze data in the "Safe Zone" far from boundaries
# This ensures the true posterior is effectively N(x, sigma)
mask = (xs > -4) & (xs < 4)
thetas = thetas[mask]
xs = xs[mask]

# Spatially Varying Bias Model
# q(theta|x) = N(x + bias, sigma)
bias_vec = np.where(xs > 0, bias_mag, -bias_mag)
ranks = stats.norm.cdf(thetas, loc=xs + bias_vec, scale=sigma_true)

# --- 2. Plotting Setup ---
# Palette (Paul Tol High Contrast)
c_ref = '#333333'    # Ground Truth (Grey)
c_marg = '#999999'   # Marginal (Light Grey)
c_pos = '#BB5566'    # Rose (x > 0)
c_neg = '#004488'    # Blue (x < 0)

fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.2))

plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'font.family': 'sans-serif' 
})

# === LEFT PANEL: Ridge Plot (Offset PDFs) ===
ax0 = axes[0]
t_grid = np.linspace(-5, 5, 500)

# Representative x values (Safe Zone)
x_samples = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
y_offsets = np.arange(len(x_samples)) * 0.5  

for i, x_val in enumerate(x_samples):
    base_y = y_offsets[i]
    
    # 1. Ground Truth Posterior: N(x, sigma)
    pdf_true = stats.norm.pdf(t_grid, loc=x_val, scale=sigma_true)
    ax0.plot(t_grid, pdf_true + base_y, color=c_ref, ls='--', lw=1.2, zorder=5)
    
    # 2. Biased Inferred Posterior
    if x_val > 0:
        bias = bias_mag
        color = c_pos
        label = 'Inferred ($x>0$)' if i == 4 else None 
    else:
        bias = -bias_mag
        color = c_neg
        label = 'Inferred ($x<0$)' if i == 1 else None 
        
    pdf_inf = stats.norm.pdf(t_grid, loc=x_val + bias, scale=sigma_true)
    ax0.plot(t_grid, pdf_inf + base_y, color=color, lw=2.0, zorder=10, label=label)
    ax0.fill_between(t_grid, base_y, pdf_inf + base_y, color=color, alpha=0.15)

ax0.plot([], [], color=c_ref, ls='--', lw=1.2, label='Ground Truth') # Dummy Legend

ax0.set_xlabel(r'$\theta$')
ax0.set_ylabel('Posterior Density (Offset)')
ax0.set_yticks([]) 
ax0.set_xlim(-5, 5)
ax0.set_ylim(-0.2, 4)
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.spines['left'].set_visible(False) 

ax0.legend(loc='upper left', frameon=False, fontsize=8)
ax0.text(-0.05, 1.0, '(a)', transform=ax0.transAxes, fontweight='bold', fontsize=10, va='top', ha='right')


# === RIGHT PANEL: Stratified Rank Histograms ===
ax1 = axes[1]
n_bins = 20
bin_edges = np.linspace(0, 1, n_bins + 1)
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

# 1. Global (Marginal) Ranks
hist_all, _ = np.histogram(ranks, bins=bin_edges, density=True)
ax1.step(bin_centers, hist_all, where='mid', color=c_ref, ls='--', lw=1.5, label='Global (all $x$)', zorder=10)

# 2. Stratified Ranks
ranks_pos = ranks[xs > 0]
hist_pos, _ = np.histogram(ranks_pos, bins=bin_edges, density=True)
ax1.step(bin_centers, hist_pos, where='mid', color=c_pos, lw=2.5, label=r'$x > 0$')

ranks_neg = ranks[xs < 0]
hist_neg, _ = np.histogram(ranks_neg, bins=bin_edges, density=True)
ax1.step(bin_centers, hist_neg, where='mid', color=c_neg, lw=2.5, label=r'$x < 0$')

ax1.axhline(1.0, color='gray', lw=0.5, alpha=0.3)
ax1.set_xlabel(r'Rank Statistic $U(\theta^*)$')
ax1.set_ylabel('Normalized Density')
ax1.set_ylim(0.0, 2.0)
ax1.set_xlim(0, 1)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax1.legend(loc='upper center', frameon=False, ncol=1, fontsize=8)
ax1.text(-0.18, 1.0, '(b)', transform=ax1.transAxes, fontweight='bold', fontsize=10, va='top', ha='right')

plt.tight_layout()
plt.savefig('figures/fig_sbc_bias.pdf', format='pdf', bbox_inches='tight')
plt.show()