import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.integrate import quad

# --- 1. Define Distributions ---
# Reference q(theta): Standard Normal N(0,1)
q_dist = stats.norm(loc=0, scale=1)

# p1: Shifted Mean -> N(1, 1)
p1_dist = stats.norm(loc=0.5, scale=1)

# p2: Broadened -> N(0, 2^2) = N(0, 4)
# Note: scipy scale is standard deviation, so scale=2
p2_dist = stats.norm(loc=0, scale=1.4)

# p3: Mixture -> 0.9*N(0,1) + 0.1*N(4,1)
class MixtureDist:
    def __init__(self, w=0.9, loc1=0, scale1=1, loc2=4, scale2=1):
        self.w = w
        self.d1 = stats.norm(loc=loc1, scale=scale1)
        self.d2 = stats.norm(loc=loc2, scale=scale2)
    
    def pdf(self, x):
        return self.w * self.d1.pdf(x) + (1 - self.w) * self.d2.pdf(x)

p3_dist = MixtureDist(w=0.95, loc1=0, scale1=1, loc2=3.0, scale2=0.5)

# --- 2. Define Divergence Metrics ---
def get_kl(p, q, limit=20):
    # KL(p||q)
    func = lambda x: p.pdf(x) * (np.log(p.pdf(x)) - np.log(q.pdf(x)))
    return quad(func, -limit, limit)[0]

def get_tv(p, q, limit=20):
    # TV(p, q)
    func = lambda x: 0.5 * np.abs(p.pdf(x) - q.pdf(x))
    return quad(func, -limit, limit)[0]

def get_chi2(p, q, limit=20):
    # Chi2(p||q)
    func = lambda x: (p.pdf(x) - q.pdf(x))**2 / q.pdf(x)
    return quad(func, -limit, limit)[0]

def get_js(p, q, limit=20):
    # JS(p, q)
    m_pdf = lambda x: 0.5 * (p.pdf(x) + q.pdf(x))
    
    func_pm = lambda x: p.pdf(x) * (np.log(p.pdf(x)) - np.log(m_pdf(x)))
    kl_pm = quad(func_pm, -limit, limit)[0]
    
    func_qm = lambda x: q.pdf(x) * (np.log(q.pdf(x)) - np.log(m_pdf(x)))
    kl_qm = quad(func_qm, -limit, limit)[0]
    
    return 0.5 * kl_pm + 0.5 * kl_qm

def get_jeffreys(p, q, limit=20):
    # J(p,q)
    return get_kl(p, q, limit) + get_kl(q, p, limit)


# --- 3. Compute Values ---
dists = [p1_dist, p2_dist, p3_dist]
dist_names = [r'$p_1$ (Shift)', r'$p_2$ (Broad)', r'$p_3$ (Mix)']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] 

# Config: (Label, Function, Scaling Factor)
# Ordered to match typical plots
metrics_config = [
    (r'$\frac{1}{2} D_{TV}(p, q)$',  lambda p: (1/2) * get_tv(p, q_dist), 1.0),
    (r'$\frac{1}{2} \chi^2(q||p)$', lambda p: 0.5 * get_chi2(q_dist, p), 1.0),
    (r'$D_{KL}(q||p)$',            lambda p: get_kl(q_dist, p), 1.0),
    (r'$4 D_{JS}(p, q)$',          lambda p: 4 * get_js(p, q_dist), 1.0),
    (r'$\frac{1}{2} D_J(p, q)$',     lambda p: 0.5 * get_jeffreys(p, q_dist), 1.0), 
    (r'$D_{KL}(p||q)$',            lambda p: get_kl(p, q_dist), 1.0),
    (r'$\frac{1}{2} \chi^2(p||q)$', lambda p: 0.5 * get_chi2(p, q_dist), 1.0),
]

results = {name: [] for name, _, _ in metrics_config}

for p in dists:
    for name, func, scale in metrics_config:
        val = func(p) * scale
        results[name].append(val)


# --- 4. Plotting ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Left Panel: Log Density Plots ---
x = np.linspace(-6, 8, 500)
axes[0].plot(x, q_dist.pdf(x), 'k--', label=r'Ref $q(\theta) = \mathcal{N}(0,1)$', lw=2)
axes[0].plot(x, p1_dist.pdf(x), color=colors[0], label=r'$p_1 = \mathcal{N}(1,1)$', lw=2)
axes[0].plot(x, p2_dist.pdf(x), color=colors[1], label=r'$p_2 = \mathcal{N}(0,4)$', lw=2)
axes[0].plot(x, [p3_dist.pdf(xi) for xi in x], color=colors[2], label=r'$p_3 = 0.9\mathcal{N}(0,1) + 0.1\mathcal{N}(4,1)$', lw=2)

axes[0].set_yscale('log')
axes[0].set_ylim(1e-4, 2.0) # Adjust to show tails without cluttering bottom
axes[0].set_title('Approximate Posteriors (Log Density)', fontsize=14)
axes[0].set_xlabel(r'$\theta$', fontsize=12)
axes[0].set_ylabel('Log Density', fontsize=12)
axes[0].legend(loc='upper right', frameon=True, fontsize=10)
axes[0].grid(True, which="both", ls="-", alpha=0.1)

# --- Right Panel: Divergences (Swapped Axes & Linear) ---
x_pos = np.arange(len(metrics_config))
metric_labels = [m[0] for m in metrics_config]

for i, p_name in enumerate(dist_names):
    # Collect values for this distribution across all metrics
    vals = [results[m_label][i] for m_label in metric_labels]
    # Plotting: X = Metric Position, Y = Value
    axes[1].plot(x_pos, vals, 'o', color=colors[i], ms=9, label=p_name, alpha=0.9)

axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(metric_labels, fontsize=12, rotation=45, ha='right')
axes[1].set_title('Divergence Measures (Linear Scale)', fontsize=14)
axes[1].set_ylabel('Value', fontsize=12)

# Add grid for readability
axes[1].grid(True, axis='y', ls="-", alpha=0.3)
axes[1].grid(True, axis='x', ls=":", alpha=0.3)

axes[1].set_ylim([1e-2, 1e1])
axes[1].set_yscale('log')

# Add legend to right plot to identify dots
axes[1].legend(title="Approximation", frameon=True)

plt.tight_layout()
plt.show()