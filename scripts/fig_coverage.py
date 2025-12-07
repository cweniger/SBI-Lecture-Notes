import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --- 1. Simulation Settings ---
np.random.seed(42)
N_sim = 10000        
sigma_true = 0.1     
prior_range = 1.0    

# Generate Ground Truth Data
thetas = np.random.uniform(-prior_range, prior_range, N_sim)
xs = np.random.normal(thetas, sigma_true)

# --- 2. Define Approximate Posteriors q(theta|x) ---
params = {
    'Exact':          {'bias': 0.0, 's': 1.0},
    'Shifted':        {'bias': 0.05, 's': 1.0},
    'Overdispersed':  {'bias': 0.0, 's': 1.3},
    'Underdispersed': {'bias': 0.0, 's': 0.8}
}

# PALETTE: Paul Tol High Contrast (Option 1: Consistent Mapping)
# We use Mustard vs Blue below the diagonal to maximize contrast.
colors = {
    'Exact': '#333333',          # Dark Grey
    'Shifted': '#DDAA33',        # Mustard
    'Overdispersed': '#BB5566',  # Rose
    'Underdispersed': '#004488'  # Dark Blue
}

# --- 3. Compute Realized Coverage ---
nominal_levels = np.linspace(0, 1, 101)
realized_coverage = {}

for name, p in params.items():
    mu = xs + p['bias']
    sigma_model = sigma_true * p['s']
    
    # z-score of truth relative to model
    z = np.abs(thetas - mu) / sigma_model
    # Prob mass inside the interval [-z, z]
    f_stats = stats.norm.cdf(z) - stats.norm.cdf(-z) 
    
    realized = [np.mean(f_stats <= c) for c in nominal_levels]
    realized_coverage[name] = realized

# --- 4. Plotting ---
fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.2))

plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'legend.fontsize': 7,
    'font.family': 'sans-serif' 
})

# === LEFT PANEL: Posterior Density Examples ===
ax0 = axes[0]
t_grid = np.linspace(-0.4, 0.5, 500)
x_vis = 0.0 

# Ground Truth
pdf_true = stats.norm.pdf(t_grid, 0, sigma_true)
ax0.plot(t_grid, pdf_true, color=colors['Exact'], ls='--', lw=2.0, label='Ground Truth', zorder=10)
ax0.fill_between(t_grid, pdf_true, color=colors['Exact'], alpha=0.05)

# Approximations
for name in ['Shifted', 'Overdispersed', 'Underdispersed']:
    p = params[name]
    pdf = stats.norm.pdf(t_grid, p['bias'], sigma_true * p['s'])
    ax0.plot(t_grid, pdf, color=colors[name], lw=2.0, label=name)
    # Increased alpha slightly for better visibility
    ax0.fill_between(t_grid, pdf, color=colors[name], alpha=0.1)

ax0.set_xlabel(r'$\theta$')
ax0.set_ylabel('Density')
ax0.set_yticks([])
ax0.set_ylim(0, 6.0)
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.legend(loc='upper right', frameon=False)
ax0.text(-0.18, 1.0, '(a)', transform=ax0.transAxes, fontweight='bold', fontsize=10, va='top', ha='right')

# === RIGHT PANEL: PP-Plot (Coverage) ===
ax1 = axes[1]

# Diagonal Reference
ax1.plot([0, 1], [0, 1], color=colors['Exact'], ls='--', lw=1.5, label='Exact', zorder=10)

for name in ['Shifted', 'Overdispersed', 'Underdispersed']:
    ax1.plot(nominal_levels, realized_coverage[name], color=colors[name], lw=2.0, label=name)

ax1.set_xlabel('Nominal Coverage ($1-\\alpha$)')
ax1.set_ylabel('Realized Coverage ($1-\\tilde{\\alpha}$)')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(True, ls='-', alpha=0.1) # Added grid for readability

# Legend (Upper Left to stay out of the way of the curves)
ax1.legend(loc='upper left', frameon=False)

## Pedagogical Annotations
#ax1.text(0.65, 0.3, 'Overconfident\n(Below Diagonal)', color=colors['Underdispersed'], 
#         fontsize=7, ha='center', fontweight='bold', alpha=0.9)
#ax1.text(0.3, 0.75, 'Underconfident\n(Above Diagonal)', color=colors['Overdispersed'], 
#         fontsize=7, ha='center', fontweight='bold', alpha=0.9)

ax1.text(-0.15, 1.0, '(b)', transform=ax1.transAxes, fontweight='bold', fontsize=10, va='top', ha='right')

plt.tight_layout()
plt.savefig('figures/fig_coverage.pdf', format='pdf', bbox_inches='tight')
plt.show()