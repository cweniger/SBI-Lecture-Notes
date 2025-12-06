import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --- 1. Simulation Settings ---
np.random.seed(42) 
N_sim = 40000        # High count for smooth histograms
L = 19               # Rank bins (0 to 19)
sigma_true = 1.0     # Ground Truth Noise

# Generate Ground Truth Data
# theta ~ U(-2, 2)
thetas = np.random.uniform(-2, 2, N_sim)
# x ~ N(theta, sigma_true)
xs = np.random.normal(thetas, sigma_true)

# --- 2. Define Approximate Posteriors q(theta|x) ---
# Mapping concepts from Fig 3.3.1 to SBC failure modes
params = {
    'Exact':          {'bias': 0.0, 'scale_fac': 1.0},
    'Shifted':        {'bias': 0.2, 'scale_fac': 1.0}, # Matches 'Shifted' (Mustard)
    'Overdispersed':  {'bias': 0.0, 'scale_fac': 1.2}, # Matches 'Broadened' (Rose)
    'Underdispersed': {'bias': 0.0, 'scale_fac': 0.8}  # New failure mode (Blue)
}

# --- 3. Compute SBC Ranks ---
ranks = {}
for name, p in params.items():
    bias = p['bias']
    sigma_model = sigma_true * p['scale_fac']
    
    # Calculate Rank Statistic (Probability Integral Transform)
    # u = CDF_model(theta_true)
    u = stats.norm.cdf(thetas, loc=xs + bias, scale=sigma_model)
    ranks[name] = u

# --- 4. Plotting ---
# PALETTE: Paul Tol High Contrast (Matches Fig 3.3.1)
colors = {
    'Exact': '#333333',          # Dark Grey
    'Shifted': '#DDAA33',        # Mustard
    'Overdispersed': '#BB5566',  # Rose
    'Underdispersed': '#004488'  # Dark Blue
}

fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.2))

# STYLE: Exact match to Fig 3.3.1
plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'font.family': 'sans-serif' 
})

# === LEFT PANEL: Posterior Densities ===
ax0 = axes[0]
x_vis = 0.0 # Visualize for observation x=0
t_grid = np.linspace(-3, 3, 500)

# 1. Exact (Reference)
p_ref = params['Exact']
pdf_ref = stats.norm.pdf(t_grid, loc=x_vis, scale=sigma_true)
ax0.plot(t_grid, pdf_ref, color=colors['Exact'], ls='--', lw=2.0, label='Exact', zorder=10)
ax0.fill_between(t_grid, pdf_ref, color=colors['Exact'], alpha=0.05)

# 2. Deviations
# Plot Underdispersed first so it sits behind the others? No, darker colors on top usually better.
# Order: Over, Shifted, Under
for name in ['Overdispersed', 'Shifted', 'Underdispersed']:
    p = params[name]
    pdf = stats.norm.pdf(t_grid, loc=x_vis + p['bias'], scale=sigma_true * p['scale_fac'])
    ax0.plot(t_grid, pdf, color=colors[name], lw=2.5, label=name)
    ax0.fill_between(t_grid, pdf, color=colors[name], alpha=0.1)

ax0.set_xlabel(r'$\theta$')
ax0.set_ylabel('Posterior Density')
ax0.set_yticks([]) # Qualitative density
ax0.set_xlim(-4, 4.)
ax0.set_ylim(0, 0.8)
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)

# Legend upper right (Inside)
ax0.legend(loc='upper right', frameon=False)
ax0.text(-0.18, 1.0, '(a)', transform=ax0.transAxes, fontweight='bold', fontsize=10, va='top', ha='right')


# === RIGHT PANEL: Rank Histograms ===
ax1 = axes[1]
n_bins = 20
bin_edges = np.linspace(0, 1, n_bins + 1)
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

# 1. Reference Line
ax1.axhline(1.0, color=colors['Exact'], ls='--', lw=1.5, alpha=0.5, label='Uniform')

# 2. Deviations
for name in ['Overdispersed', 'Shifted', 'Underdispersed']:
    # Compute histogram density
    hist, _ = np.histogram(ranks[name], bins=bin_edges, density=True)
    # Plot stepped line
    ax1.step(bin_centers, hist, where='mid', color=colors[name], lw=2.5)

ax1.set_xlabel(r'Rank Statistic $U(\theta^*)$')
ax1.set_ylabel('Normalized Density')
ax1.set_ylim(0.0, 2.5)
ax1.set_xlim(0, 1)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# 3. Direct Annotations (Pedagogical alignment)
# Underdispersed -> U-Shape (Blue)
ax1.text(0.5, 0.25, 'Underdispersed\n(U-shape)', color=colors['Underdispersed'], 
         ha='center', va='bottom', fontsize=7, fontweight='bold')

# Overdispersed -> Inverted U (Rose)
ax1.text(0.5, 1.45, 'Overdispersed\n(Inverted U)', color=colors['Overdispersed'], 
         ha='center', va='bottom', fontsize=7, fontweight='bold')

# Shifted -> Slope (Mustard)
ax1.text(0.85, 1.8, 'Shifted\n(Bias)', color=colors['Shifted'], 
         ha='center', va='bottom', fontsize=7, fontweight='bold')

ax1.text(-0.15, 1.0, '(b)', transform=ax1.transAxes, fontweight='bold', fontsize=10, va='top', ha='right')

plt.tight_layout()
plt.savefig('figure_3_4_1_sbc.pdf', format='pdf', bbox_inches='tight')
plt.show()