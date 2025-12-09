import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --- 1. Simulation Settings ---
np.random.seed(43)
N_sim = 20000        # Number of simulations
L_samples = 1000     # Posterior samples for finite-sample rank
sigma_true = 1.0

# Generate Data
# theta ~ N(0, 1)
thetas = np.random.normal(0, 1, N_sim)
# Two observations: x1, x2 ~ N(theta, 1)
x1 = np.random.normal(thetas, sigma_true)
x2 = np.random.normal(thetas, sigma_true)

# --- 2. Define Posteriors ---

# A. Ground Truth (Full Info: x1, x2)
# Precision = 1 (prior) + 2 (data) = 3
mu_full = (x1 + x2) / 3.0
sigma_full = np.sqrt(1.0 / 3.0)

# B. Lossy Summary (Only x1)
# Precision = 1 + 1 = 2
mu_lossy = x1 / 2.0
sigma_lossy = np.sqrt(1.0 / 2.0)

# C. Overdispersed (Full Mean, Lossy Variance)
mu_over = mu_full
sigma_over = sigma_lossy

# --- 3. Compute Ranks for Diagnostics ---
# We need to simulate the "Rank" of the true theta against the posterior samples.
# Rank = P( TestFunc(theta_samples) < TestFunc(theta_true) )

# Pre-compute Full Log-Likelihood for Truth
# log p(x1, x2 | theta) proportional to - (x1-theta)^2 - (x2-theta)^2
def log_lik_full(t, d1, d2):
    return - (d1 - t)**2 - (d2 - t)**2

# 1. Lossy Model - HPDR Diagnostic (Ordering by q density)
# For Normal q, this is equivalent to ordering by proximity to mean: -|theta - mu|
# Since Lossy is a valid posterior, this must be Uniform (Diagonal).
# u = CDF(theta_true) for the N(mu_lossy, sigma_lossy)
rank_lossy_hpdr = stats.norm.cdf(thetas, loc=mu_lossy, scale=sigma_lossy)
# Convert to "Central Interval Coverage" for PP-plot (distance from median)
# abs(2*u - 1) transforms U[0,1] to "centered coverage" U[0,1]
# But standard PP-plots use the rank directly.
# Let's use the "Symmetric Interval" coverage as in Fig 3.4.4 logic
z_lossy = np.abs(thetas - mu_lossy) / sigma_lossy
cov_lossy_hpdr = stats.norm.cdf(z_lossy) - stats.norm.cdf(-z_lossy)

# 2. Overdispersed Model - HPDR Diagnostic
# Miscalibrated -> Should deviate
z_over = np.abs(thetas - mu_over) / sigma_over
cov_over_hpdr = stats.norm.cdf(z_over) - stats.norm.cdf(-z_over)

# 3. Lossy Model - Likelihood Diagnostic
# Ordering function: Full Likelihood p(x1, x2 | theta)
# We compare Lik(theta_true) vs Lik(theta_samples_from_lossy)
# Generate samples from Lossy Posterior (N_sim, L_samples)
# To save memory, we can do this analytically or via smaller batches.
# Simulation approach:
rank_lossy_lik = []
batch_size = 1000
for i in range(0, N_sim, batch_size):
    end = min(i + batch_size, N_sim)
    n_batch = end - i
    
    # Truth and Data for batch
    t_true = thetas[i:end]
    d1 = x1[i:end]
    d2 = x2[i:end]
    
    # Samples from Lossy q(theta|x1)
    # shape (n_batch, L_samples)
    samples = np.random.normal(mu_lossy[i:end, None], sigma_lossy, size=(n_batch, 100))
    
    # Evaluate Test Function (Full Likelihood)
    score_true = log_lik_full(t_true, d1, d2)            # (n_batch,)
    score_samples = log_lik_full(samples, d1[:,None], d2[:,None]) # (n_batch, L)
    
    # Rank: Fraction of samples with LOWER likelihood than truth
    # If truth explains full data better than lossy samples, truth has HIGH score.
    # So Rank should be high.
    r = np.mean(score_samples < score_true[:, None], axis=1)
    rank_lossy_lik.extend(r)

rank_lossy_lik = np.array(rank_lossy_lik)

# --- 4. Plotting ---
fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.2))

plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'legend.fontsize': 7,
    'font.family': 'sans-serif' 
})

colors = {'Lossy': '#DDAA33', 'Over': '#BB5566', 'True': '#333333'}

# === LEFT PANEL: Ridge Plot (Same as Fig 3.4.3) ===
ax0 = axes[0]
t_grid = np.linspace(-3, 3, 500)
viz_indices = np.argsort(thetas)[np.linspace(2000, 18000, 6, dtype=int)]
y_offsets = np.arange(len(viz_indices)) * 0.6

for i, idx in enumerate(viz_indices):
    base_y = y_offsets[i]
    # Truth
    pdf_true = stats.norm.pdf(t_grid, loc=mu_full[idx], scale=sigma_full)
    ax0.plot(t_grid, pdf_true + base_y, color=colors['True'], ls='--', lw=1.2, zorder=10)
    # Overdispersed
    pdf_over = stats.norm.pdf(t_grid, loc=mu_over[idx], scale=sigma_over)
    ax0.plot(t_grid, pdf_over + base_y, color=colors['Over'], lw=2.0, zorder=5)
    # Lossy
    pdf_lossy = stats.norm.pdf(t_grid, loc=mu_lossy[idx], scale=sigma_lossy)
    ax0.plot(t_grid, pdf_lossy + base_y, color=colors['Lossy'], lw=2.0, zorder=8)
    ax0.fill_between(t_grid, base_y, pdf_lossy + base_y, color=colors['Lossy'], alpha=0.2)

ax0.set_xlabel(r'$\theta$')
ax0.set_ylabel('Posterior Density (Offset)')
ax0.set_yticks([])
ax0.set_xlim(-3, 3)
ax0.set_ylim(-0.2, 4.5)
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.spines['left'].set_visible(False)
# Dummy Legend
ax0.plot([], [], color=colors['Lossy'], lw=2, label='Lossy Summary')
ax0.plot([], [], color=colors['Over'], lw=2, label='Overdispersed')
ax0.legend(loc='upper left', frameon=False)
ax0.text(-0.05, 1.0, '(a)', transform=ax0.transAxes, fontweight='bold', fontsize=10, va='top', ha='right')
#ax0.set_title('Posterior Width vs Information', fontsize=9)

# === RIGHT PANEL: Coverage Comparison ===
ax1 = axes[1]
nominal = np.linspace(0, 1, 100)

# 1. Diagonal Ref
ax1.plot([0, 1], [0, 1], color=colors['True'], ls=':', lw=1.0)

# 2. Lossy HPDR (Solid Mustard) -> Diagonal
# Calculate ECDF of coverage values
realized_lossy_hpdr = [np.mean(cov_lossy_hpdr <= alpha) for alpha in nominal]
ax1.plot(nominal, realized_lossy_hpdr, color=colors['Lossy'], lw=2.5, label='Lossy (HPDR Test)')

# 3. Overdispersed HPDR (Solid Rose) -> Underconfident (Above Diag)
realized_over_hpdr = [np.mean(cov_over_hpdr <= alpha) for alpha in nominal]
ax1.plot(nominal, realized_over_hpdr, color=colors['Over'], lw=2.5, label='Overdisp. (HPDR Test)')

# 4. Lossy LIKELIHOOD Rank (Dashed Mustard) -> Deviates
# Rank distribution is non-uniform. We plot the PP-plot of the rank.
# Sorted ranks vs Uniform
ranks_sorted = np.sort(rank_lossy_lik)
uniform_grid = np.linspace(0, 1, len(ranks_sorted))
# "Realized" is the y-axis (Empirical CDF), "Nominal" is x-axis
ax1.plot(uniform_grid, ranks_sorted, color='#004488', ls='-', lw=2.5, label='Lossy (Likelihood Test)')

ax1.set_xlabel('Nominal Level')
ax1.set_ylabel('Realized Level')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.grid(True, ls='-', alpha=0.1)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax1.legend(loc='upper left', frameon=False, fontsize=7)
ax1.text(-0.15, 1.0, '(b)', transform=ax1.transAxes, fontweight='bold', fontsize=10, va='top', ha='right')
#ax1.set_title('Diagnostic Sensitivity', fontsize=9)

# Annotation for the key insight
#ax1.text(0.65, 0.25, 'Likelihood test\ndetects info loss!', color=colors['Lossy'], 
#         fontsize=7, fontweight='bold', ha='center', alpha=0.9)

plt.tight_layout()
plt.savefig('figures/fig_coverage_likelihood_based.pdf', format='pdf', bbox_inches='tight')
plt.show()