import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import logsumexp

# --- 1. Settings & Distribution Definitions ---
np.random.seed(43) 
N_q = 64 # Number of posterior samples

# --- Define Custom Mixture Class ---
class GaussianMixture:
    def __init__(self, weights, means, covs):
        self.weights = np.array(weights)
        self.weights /= self.weights.sum() # Normalize
        self.means = [np.array(m) for m in means]
        self.covs = [np.array(c) for c in covs]
        self.precs = [np.linalg.inv(c) for c in covs] # Precompute precisions
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

    def score(self, x):
        """Computes the gradient of log p(x) with respect to x."""
        x = np.atleast_2d(x)
        n_samples = x.shape[0]
        
        numerator = np.zeros_like(x)
        denominator = np.zeros(n_samples)
        
        for w, m, c, P in zip(self.weights, self.means, self.covs, self.precs):
            # p_k(x)
            dens = w * stats.multivariate_normal.pdf(x, mean=m, cov=c)
            dens = np.atleast_1d(dens)
            
            denominator += dens
            
            # Gradient of log p_k(x) is -P(x - mu)
            diff = x - m
            grad_log_pk = -np.einsum('ij,nj->ni', P, diff)
            
            numerator += dens[:, None] * grad_log_pk
            
        return numerator / (denominator[:, None] + 1e-15)

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

# Score Components (Nabla log p)
def ord_score_0(t): return dist_p.score(t)[:, 0]
def ord_score_1(t): return dist_p.score(t)[:, 1] # NEW

# Log Ratio
def ord_log_ratio(t): return dist_q.logpdf(t) - dist_p.logpdf(t)


# --- 3. Panel Configuration ---
# Row 1: Dist, theta1, theta2, log q
# Row 2: log(q/p), nabla_1, nabla_2, log p
panels = [
    # Row 1
    ('Raw Distribution', None),
    (r'Order: $\theta_1$', ord_theta1),
    (r'Order: $\theta_2$', ord_theta2),
    (r'Order: $\log q(\boldsymbol{\theta} \mid \mathbf{x})$', ord_log_q),
    
    # Row 2
    (r'Order: $\log \frac{q(\boldsymbol{\theta}\mid \mathbf{x})}{\pi(\boldsymbol{\theta} \mid \mathbf{x})}$', ord_log_ratio),
    (r'Order: $\nabla_{\boldsymbol{\theta}_1} \log \pi(\boldsymbol{\theta} \mid \mathbf{x})$', ord_score_0),
    (r'Order: $\nabla_{\boldsymbol{\theta}_2} \log \pi(\boldsymbol{\theta} \mid \mathbf{x})$', ord_score_1), # New
    (r'Order: $\log \pi(\boldsymbol{\theta} \mid \mathbf{x})$', ord_log_p),
]

# --- 4. Plotting Helper ---
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
    #print(sorted_scores.min(), sorted_scores.max())
    L = len(sorted_scores)
    colors = plt.cm.viridis(np.linspace(0, 1, L))
    ax.contour(X, Y, Z, levels=sorted_scores, colors=colors,
               alpha=0.4, linewidths=0.6, zorder=1)
    
    # 2. Contour for True Sample
    ax.contour(X, Y, Z, levels=[score_true], colors='#BB5566', 
               linewidths=2.0, zorder=10)

# --- 5. Main Plot ---
fig, axes = plt.subplots(2, 4, figsize=(6.3, 3.2), sharex=True, sharey=True)
axes = axes.flatten()
plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0.01, right=0.99, top=0.99, bottom=0.01)

bounds = [-4, 4, -4, 4]
props = dict(boxstyle='square,pad=0.2', facecolor='white', alpha=0.9, edgecolor='none')

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
        
        # Safe LaTeX String
        label_text_p = r"True $p(\boldsymbol{\theta} \mid \mathbf{x})$ (blue)"
        label_text_q = r"Learned $q(\boldsymbol{\theta} \mid \mathbf{x})$ (dash)"
        
        ax.text(0.05, 0.96, label_text_p, transform=ax.transAxes, 
                fontsize=8, color='#004488', fontweight='bold', verticalalignment='top', bbox=props, zorder=100)
        ax.text(0.05, 0.85, label_text_q, transform=ax.transAxes, 
                fontsize=8, color='black', verticalalignment='top', bbox=props, zorder=100)
        
        ax.set_facecolor('#fdfdfd')
        
    else:
        draw_contours(ax, func, theta_true, thetas_q, bounds)
        ax.text(0.03, 0.97, title, transform=ax.transAxes, 
                fontsize=8, verticalalignment='top', bbox=props, zorder=100)
        
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_xticks([])
    ax.set_yticks([])
    
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color('#cccccc')

plt.savefig('figures/fig_grd_illustration.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.show()