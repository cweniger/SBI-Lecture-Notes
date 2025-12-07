import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.integrate import quad

# --- 1. Define Distributions ---
q_dist = stats.norm(loc=0, scale=1) 
p1_dist = stats.norm(loc=0.5, scale=1) 
p2_dist = stats.norm(loc=0, scale=1.4) 

class MixtureDist:
    def __init__(self, w=0.9, loc1=0, scale1=1, loc2=4, scale2=1):
        self.w = w
        self.d1 = stats.norm(loc=loc1, scale=scale1)
        self.d2 = stats.norm(loc=loc2, scale=scale2)
    def pdf(self, x):
        return self.w * self.d1.pdf(x) + (1 - self.w) * self.d2.pdf(x)

p3_dist = MixtureDist(w=0.95, loc1=0, scale1=1, loc2=3.0, scale2=0.5)

# --- 2. Metrics ---
def get_kl(p, q, limit=20):
    func = lambda x: p.pdf(x) * (np.log(p.pdf(x)) - np.log(q.pdf(x)))
    return quad(func, -limit, limit)[0]

def get_tv(p, q, limit=20):
    func = lambda x: 0.5 * np.abs(p.pdf(x) - q.pdf(x))
    return quad(func, -limit, limit)[0]

def get_chi2(p, q, limit=20):
    func = lambda x: (p.pdf(x) - q.pdf(x))**2 / q.pdf(x)
    return quad(func, -limit, limit)[0]

def get_js(p, q, limit=20):
    m_pdf = lambda x: 0.5 * (p.pdf(x) + q.pdf(x))
    return 0.5 * quad(lambda x: p.pdf(x)*(np.log(p.pdf(x))-np.log(m_pdf(x))), -limit, limit)[0] + \
           0.5 * quad(lambda x: q.pdf(x)*(np.log(q.pdf(x))-np.log(m_pdf(x))), -limit, limit)[0]

def get_jeffreys(p, q, limit=20):
    return get_kl(p, q, limit) + get_kl(q, p, limit)

# --- 3. Compute ---
dists = [p1_dist, p2_dist, p3_dist]
dist_names = ['Shifted', 'Broadened', 'Heavy-tailed']

# PALETTE: Paul Tol High Contrast (Optimized for Slides/Projectors)
colors = ['#DDAA33', '#BB5566', '#004488'] # Mustard, Rose, Dark Blue
markers = ['o', 's', '^'] 

metrics_config = [
    (r'$\frac{1}{2} D_{TV}(p, q)$',  lambda p: (1/2) * get_tv(p, q_dist), 1.0),
    (r'$\frac{1}{2} D_{\chi^2}(q||p)$', lambda p: 0.5 * get_chi2(q_dist, p), 1.0),
    (r'$D_{KL}(q||p)$',            lambda p: get_kl(q_dist, p), 1.0),
    (r'$4 D_{JS}(p, q)$',          lambda p: 4 * get_js(p, q_dist), 1.0),
    (r'$\frac{1}{2} D_J(p, q)$',     lambda p: 0.5 * get_jeffreys(p, q_dist), 1.0), 
    (r'$D_{KL}(p||q)$',            lambda p: get_kl(p, q_dist), 1.0),
    (r'$\frac{1}{2} D_{\chi^2}(p||q)$', lambda p: 0.5 * get_chi2(p, q_dist), 1.0),
]

results = {name: [] for name, _, _ in metrics_config}
for p in dists:
    for name, func, scale in metrics_config:
        results[name].append(func(p) * scale)

# --- 4. Plotting ---
fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.2))

# STYLE
plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'font.family': 'sans-serif' 
})

# === LEFT PANEL ===
x = np.linspace(-6, 8, 500)
# Reference
axes[0].plot(x, q_dist.pdf(x), color='#333333', linestyle='--', label='Reference', lw=2, zorder=10)
axes[0].fill_between(x, q_dist.pdf(x), color='#333333', alpha=0.05)

for i, p in enumerate(dists):
    y = [p.pdf(xi) for xi in x] if isinstance(p, MixtureDist) else p.pdf(x)
    axes[0].plot(x, y, color=colors[i], label=dist_names[i], lw=2.5) 
    axes[0].fill_between(x, y, color=colors[i], alpha=0.1)

axes[0].set_yscale('log')
axes[0].set_ylim(1e-3, 2.0)
axes[0].set_xlim(-5, 10)
axes[0].set_xlabel(r'$\theta$')
axes[0].set_ylabel('Probability density')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# ADJUSTMENT 1: Moved legend to upper left to avoid overlap with heavy tails
axes[0].legend(loc='upper right', frameon=False) 

# Tag (a)
axes[0].text(-0.18, 1.0, '(a)', transform=axes[0].transAxes, 
             fontsize=10, fontweight='bold', va='top', ha='right')


# === RIGHT PANEL ===
x_pos = np.arange(len(metrics_config))
metric_labels = [m[0] for m in metrics_config]
y_min_log = 1e-2 

for i, p_name in enumerate(dist_names):
    vals = [results[m_label][i] for m_label in metric_labels]
    # Lollipops
    axes[1].vlines(x=x_pos + (i-1)*0.2, ymin=y_min_log, ymax=vals, color=colors[i], alpha=0.5, lw=2)
    axes[1].plot(x_pos + (i-1)*0.2, vals, marker=markers[i], linestyle='None', 
                 color=colors[i], ms=7, label=p_name, alpha=1.0, markeredgecolor='white', markeredgewidth=0.5)

# ADJUSTMENT 2: Removed offset. Ticks are now mathematically centered on the groups.
axes[1].set_xticks(x_pos) 
axes[1].set_xticklabels(metric_labels, rotation=45, ha='center')

axes[1].set_ylabel('Divergence value')
axes[1].set_ylim([1e-2, 3e0]) 
axes[1].set_yscale('log')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].grid(True, axis='y', ls="-", alpha=0.2, color='gray')
axes[1].legend(loc='upper left', frameon=False)

# Tag (b)
axes[1].text(-0.15, 1.0, '(b)', transform=axes[1].transAxes, 
             fontsize=10, fontweight='bold', va='top', ha='right')

plt.tight_layout()
plt.savefig('figures/fig_divergences.pdf', format='pdf', bbox_inches='tight')
plt.show()