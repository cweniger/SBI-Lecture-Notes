import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import logsumexp

# --- 1. Settings & Distribution Definitions ---
np.random.seed(43) 
N_q = 50 # Number of posterior samples

# --- Define Custom Mixture Class ---
class GaussianMixture:
    def __init__(self, weights, means, covs):
        self.weights = np.array(weights)
        self.weights /= self.weights.sum() # Normalize
        self.means = [np.array(m) for m in means]
        self.covs = [np.array(c) for c in covs]
        self.n_components = len(weights)
        
    def pdf(self, x):
        x = np.atleast_2d(x)
        probs = np.zeros(x.shape[0])
        for w, m, c in zip(self.weights, self.means, self.covs):
            probs += w * stats.multivariate_normal.pdf(x, mean=m, cov=c)
        return probs

    def logpdf(self, x):
        x = np.atleast_2d(x)
        component_log_pdfs = []
        for w, m, c in zip(self.weights, self.means, self.covs):
            log_w = np.log(w)
            log_p = stats.multivariate_normal.logpdf(x, mean=m, cov=c)
            component_log_pdfs.append(log_w + log_p)
        return logsumexp(np.column_stack(component_log_pdfs), axis=1)

    def rvs(self, size=1):
        choices = np.random.choice(self.n_components, size=size, p=self.weights)
        samples = np.zeros((size, 2))
        for i in range(size):
            comp_idx = choices[i]
            samples[i] = stats.multivariate_normal.rvs(
                mean=self.means[comp_idx], cov=self.covs[comp_idx])
        return samples

# --- PARAMETERS ---
# True Posterior p (Bimodal Mixture)
p_weights = [0.7, 0.3]
p_means   = [[-0.5, -1.0], [0.5, 1.0]]
p_covs    = [[[0.6, 0.0], [0.0, 0.6]], 
             [[0.6, 0.0], [0.0, 0.6]]]

# Learned Posterior q (Single Broad Mode)
q_mean = [0.0, 0.0]
q_cov  = [[1.5, 1.0], [1.0, 1.5]]

# Initialize
dist_p = GaussianMixture(p_weights, p_means, p_covs)
dist_q = stats.multivariate_normal(mean=q_mean, cov=q_cov)

# Generate Samples
theta_true = dist_p.rvs(1).flatten()
thetas_q = dist_q.rvs(N_q)

# --- 2. Define Ordering Functions ---
def wrap(x): return np.atleast_1d(x)

def ord_theta1(t): return t[:, 0]
def ord_theta2(t): return t[:, 1]
def ord_log_q(t):  return dist_q.logpdf(t)
def ord_log_p(t):  return dist_p.logpdf(t)
def ord_dist_origin(t): return -np.linalg.norm(t, axis=1) 
def ord_proj_diag(t): return (t[:,0] + t[:,1]) / np.sqrt(2) 
def ord_proj_anti(t): return (t[:,0] - t[:,1]) / np.sqrt(2)
def ord_random(t): 
    np.random.seed(1)
    w = np.random.randn(2)
    return t.dot(w)

panels = [
    ('Raw Distribution', None),
    (r'Order: $\theta_1$', ord_theta1),
    (r'Order: $\theta_2$', ord_theta2),
    (r'Order: $\log q(\theta)$', ord_log_q),
    (r'Order: $\log p(\theta)$', ord_log_p),
    (r'Order: Radial', ord_dist_origin),
    (r'Order: Diagonal (+)', ord_proj_diag),
    (r'Order: Diagonal (-)', ord_proj_anti),
]

# --- 3. Plotting Helper ---
def draw_contours(ax, ord_func, t_true, t_samples, bounds):
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[2], bounds[3], 100)
    X, Y = np.meshgrid(x, y)
    XY = np.column_stack([X.ravel(), Y.ravel()])
    
    Z = ord_func(XY).reshape(X.shape)
    
    val_true = ord_func(t_true.reshape(1, -1))
    score_true = wrap(val_true)[0]
    
    scores_q = ord_func(t_samples)
    scores_q = wrap(scores_q)
    sorted_scores = np.sort(scores_q)
    
    # 1. Contours for Learned Samples
    ax.contour(X, Y, Z, levels=sorted_scores, cmap='viridis', 
               alpha=0.4, linewidths=0.6, zorder=1)
    
    # 2. Contour for True Sample
    ax.contour(X, Y, Z, levels=[score_true], colors='#BB5566', 
               linewidths=2.0, zorder=10)

# --- 4. Main Plot ---
# Size: 6.3 inches width corresponds to typical text width on A4 with margins
# Height 3.2 makes individual panels roughly square/compact
fig, axes = plt.subplots(2, 4, figsize=(6.3, 3.2), sharex=True, sharey=True)
axes = axes.flatten()
plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0.01, right=0.99, top=0.99, bottom=0.01)

bounds = [-4, 4, -4, 4]

# Styling box for internal titles
props = dict(boxstyle='square,pad=0.2', facecolor='white', alpha=0.85, edgecolor='none')

for i, (title, func) in enumerate(panels):
    ax = axes[i]
    
    # Scatter points
    ax.scatter(thetas_q[:, 0], thetas_q[:, 1], c='gray', s=5, alpha=0.8, zorder=5)
    ax.scatter(theta_true[0], theta_true[1], c='#BB5566', marker='*', s=80, zorder=15, edgecolors='white', linewidth=0.5)

    if i == 0:
        # Panel 1: Distributions Reference
        x = np.linspace(bounds[0], bounds[1], 100)
        y = np.linspace(bounds[2], bounds[3], 100)
        X, Y = np.meshgrid(x, y)
        pos_flat = np.column_stack([X.ravel(), Y.ravel()])
        
        pdf_q_val = dist_q.pdf(pos_flat).reshape(X.shape)
        pdf_p_val = dist_p.pdf(pos_flat).reshape(X.shape)
        
        ax.contour(X, Y, pdf_q_val, colors='black', linestyles='--', alpha=0.6, linewidths=0.8)
        ax.contour(X, Y, pdf_p_val, colors='#004488', linestyles='-', alpha=0.5, linewidths=0.8)
        
        # Manually construct "Legend" inside text box
        ax.text(0.03, 0.97, 'True p (blue)\nLearned q (dash)', transform=ax.transAxes, 
                fontsize=7, verticalalignment='top', bbox=props)
        ax.set_facecolor('#fafafa')
        
    else:
        draw_contours(ax, func, theta_true, thetas_q, bounds)
        # Title inside panel
        ax.text(0.03, 0.97, title, transform=ax.transAxes, 
                fontsize=7, verticalalignment='top', bbox=props)
        
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Remove outer spines for cleaner look
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color('#cccccc')

plt.savefig('fig_grd_illustration.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.show()