import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --- 1. Simulation Settings ---
np.random.seed(42)
N_sim = 1000        # Number of simulations (Finite N)
L_samples = 100     # Number of posterior samples per sim (Finite L)

# True Posterior p(theta) = N(0, 1)
# Generate N Ground Truth parameters
theta_true = np.random.normal(0, 1, N_sim)

# --- 2. Define Pathological Posteriors ---
# q(theta) = 0.5*N(0,1) + 0.5*N(2*theta0, 1)
cases = [
    {'theta0': 0.5,  'label': 'Shifted (Small)', 'color': '#DDAA33'}, # Mustard
    {'theta0': 1.0,  'label': 'Shifted (Medium)',  'color': '#BB5566'}, # Rose
    {'theta0': 1.5, 'label': 'Shifted (Large)',    'color': '#004488'}  # Blue
]

# --- 3. Run Finite-Sample Simulation ---
# We compute the "Percentile of the Truth" based on symmetric intervals.
# The interval is defined by distance from theta0: d = |theta - theta0|
# We check what fraction of posterior samples have a smaller distance than the truth.

results = {}

for case in cases:
    t0 = case['theta0']
    
    # A. Generate Posterior Samples (N x L)
    # Mixture of two normals
    # 1. Decide component (0 or 1) for each sample
    components = np.random.randint(0, 2, size=(N_sim, L_samples))
    
    # 2. Draw from N(0,1) or N(2t0, 1) based on component
    # Base samples
    raw_samples = np.random.normal(0, 1, size=(N_sim, L_samples))
    # Shift component 1 to 2*t0
    theta_post = raw_samples + (components * 2 * t0)
    
    # B. Compute "Coverage Scores"
    # The statistic defining the Credible Region is distance from center t0.
    # dist = |theta - t0|
    dist_true = np.abs(theta_true - t0)
    dist_post = np.abs(theta_post - t0)
    
    # C. Calculate Rank / Percentile
    # For each sim, fraction of posterior samples with distance < true distance
    # This is effectively the "1 - alpha" level at which truth would be excluded.
    # Broadcasting: (N, 1) compared to (N, L)
    # Using mean over L axis gives the empirical CDF value
    ranks = np.mean(dist_post < dist_true[:, None], axis=1)
    
    results[case['label']] = ranks

# --- 4. Plotting ---
fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.2))

plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'legend.fontsize': 7,
    'font.family': 'sans-serif' 
})

# === LEFT PANEL: Densities (Visualizing the Problem) ===
ax0 = axes[0]
t_grid = np.linspace(-3, 7, 500)

# Truth
pdf_true = stats.norm.pdf(t_grid, 0, 1)
ax0.plot(t_grid, pdf_true, color='#333333', ls='--', lw=2.0, label='True Posterior', zorder=10)
ax0.fill_between(t_grid, pdf_true, color='#333333', alpha=0.05)

# Approximations
for case in cases:
    t0 = case['theta0']
    pdf_q = 0.5 * stats.norm.pdf(t_grid, 0, 1) + 0.5 * stats.norm.pdf(t_grid, 2*t0, 1)
    ax0.plot(t_grid, pdf_q, color=case['color'], lw=2.0, label=case['label'])
    ax0.fill_between(t_grid, pdf_q, color=case['color'], alpha=0.1)

ax0.set_xlabel(r'$\theta$')
ax0.set_ylabel('Density')
ax0.set_yticks([])
ax0.set_xlim(-3, 7)
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.legend(loc='upper right', frameon=False)
ax0.text(-0.1, 1.0, '(a)', transform=ax0.transAxes, fontweight='bold', fontsize=10, va='top', ha='right')
#ax0.set_title('Shape-Distorted Posteriors', fontsize=9)


# === RIGHT PANEL: Simulated Coverage (The Failure) ===
ax1 = axes[1]

# Diagonal Reference
ax1.plot([0, 1], [0, 1], color='#333333', ls=':', lw=1.0, label='Exact', zorder=0)

# Sort and plot the empirical ranks (ECDF)
# This is the PP-plot: sorted_ranks vs uniform_grid
nominal_grid = np.linspace(0, 1, N_sim)

for i, case in enumerate(cases):
    ranks = results[case['label']]
    ranks_sorted = np.sort(ranks)
    
    # Plotting the finite-sample ECDF
    # Thicker lines to make them visible on top of each other
    ls = ['-', '--', '-.'][i] # Vary linestyle to show they overlap
    ax1.plot(nominal_grid, ranks_sorted, color=case['color'], ls=ls, lw=2.5, label=case['label'], alpha=0.9)

ax1.set_xlabel('Nominal Coverage ($1-\\alpha$)')
ax1.set_ylabel('Realized Coverage ($1-\\tilde{\\alpha}$)')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(True, ls='-', alpha=0.1)

ax1.text(-0.15, 1.0, '(b)', transform=ax1.transAxes, fontweight='bold', fontsize=10, va='top', ha='right')
#ax1.set_title(f'Diagnostic Blind Spot\n(Simulated N={N_sim}, L={L_samples})', fontsize=9)

## Annotation
#ax1.text(0.6, 0.25, 'Simulated coverage\nmatches perfectly!', 
#         ha='center', va='center', fontsize=8, fontweight='bold', color='#333333')

plt.tight_layout()
plt.savefig('figures/fig_coverage_shape_distortion.pdf', format='pdf', bbox_inches='tight')
plt.show()