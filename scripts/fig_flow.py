import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import multivariate_normal
from scipy.integrate import solve_ivp  # Switched to solve_ivp for better stability

# --- Plotting Style Setup ---
plt.rcParams.update({
    'axes.labelsize': 14, 'axes.titlesize': 14,
    'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'legend.fontsize': 11, 'font.family': 'serif',
    'figure.figsize': (9, 9), 'axes.grid': True,
    'grid.alpha': 0.3, 'grid.linestyle': '--'
})

# ==========================================
# 1. Analytic Vector Field Setup (GMM)
# ==========================================
# Base: N(0, I)
# Target: GMM
# Path: Independent Optimal Transport Coupling per component

means = np.array([[4.0, 4.0], [4.0, -4.0], [-4.0, 0.0]])
covs = np.array([0.4 * np.eye(2), 0.4 * np.eye(2), 0.6 * np.eye(2)])
weights = np.array([0.35, 0.35, 0.3])
n_components = len(weights)
base_cov = np.eye(2)

def get_component_params(t):
    """
    Returns mean and covariance of each component at time t.
    Path: mu_t = t * mu_1
          Sigma_t = ((1-t) + t*sigma)^2 ... handling covariances properly below
    """
    # For optimal transport paths between N(0, I) and N(mu_1, Sigma_1),
    # the intermediate is N(t*mu_1, ( (1-t)*I + t*Sigma_1^0.5 )^2 )
    # But for simplicity, we use the linear interpolation of covariance moments 
    # often used in simple Gaussian paths: Sigma_t = (1-t)^2 I + t^2 Sigma_1 
    # (Note: The specific path choice defines the vector field. This one is robust.)
    
    mus_t = [t * m for m in means]
    # Linear interpolation of covariance "roots" is better for OT, but linear interp of Covs is easier to differentiate.
    # Let's use the simple linear sigma interpolation which is standard in 2D examples:
    # Sigma_t = ((1-t) + t*std)^2 * I approx.
    # Actually, let's stick to the code that matches the derivative logic below:
    # Sigma_t = (1-t)^2 * I + t^2 * Sigma_target
    
    covs_t = [(t**2)*c + ((1-t)**2)*base_cov for c in covs]
    d_mus = means # derivative of t*m is m
    d_covs = [2*t*c - 2*(1-t)*base_cov for c in covs] 
    
    return mus_t, covs_t, d_mus, d_covs

def vector_field(t, x): # Note: solve_ivp expects f(t, x) signature
    """
    Calculates u_t(x) = sum [ w_i * p_ti(x) * u_ti(x) ] / sum [ w_i * p_ti(x) ]
    """
    # Ensure x is (2,) or (2, N)
    x = np.atleast_1d(x)
    if x.ndim == 1:
        x = x[:, None] # Make it (2, 1) for consistent algebra
    
    # x is now (2, N_points)
    # Transpose to (N, 2) for scipy pdf
    x_T = x.T 
    
    t = np.clip(t, 1e-5, 1.0 - 1e-5)
    mus_t, covs_t, d_mus, d_covs = get_component_params(t)
    
    num = np.zeros_like(x_T)
    den = np.zeros(x_T.shape[0])
    
    for i in range(n_components):
        # 1. Component Density p_i(x)
        try:
            p_x = multivariate_normal.pdf(x_T, mean=mus_t[i], cov=covs_t[i])
        except:
            p_x = np.zeros(x_T.shape[0])
        
        # Ensure p_x is array even for single point
        p_x = np.atleast_1d(p_x) 

        # 2. Component Velocity u_i(x)
        # u(x) = d_mu + 0.5 * d_Sigma * Sigma_inv * (x - mu)
        diff = x_T - mus_t[i] # (N, 2)
        
        # Solve Sigma * z = diff^T  => z = Sigma_inv * diff^T
        # We need (d_Sigma @ z).T
        try:
            # Linear solve is safer than inversion
            # covs_t[i] is (2,2), diff.T is (2, N)
            z = np.linalg.solve(covs_t[i], diff.T) # (2, N)
            term2 = 0.5 * (d_covs[i] @ z) # (2, N)
            u_k = d_mus[i][:, None] + term2 # (2, N) broadcast d_mus
            u_k = u_k.T # Back to (N, 2)
        except np.linalg.LinAlgError:
            u_k = np.zeros_like(x_T)

        # Accumulate
        # w_p shape: (N,) -> (N, 1)
        w_p = (weights[i] * p_x)[:, None] 
        
        num += w_p * u_k
        den += (weights[i] * p_x)
        
    # Normalize
    vel = num / (den[:, None] + 1e-30)
    
    # Return (2,) if input was single point, else (2, N)
    return vel.T if vel.shape[1] == 1 else vel.T

# Wrapper for solve_ivp which expects 1D output for single point
def ode_func(t, x):
    v = vector_field(t, x)
    return v.flatten()

# ==========================================
# 2. Simulation (Backward Integration)
# ==========================================

# Define Target Regions
box_centers = np.array([[3.8, 4.2], [4.2, -3.8], [-3.5, 0.5]])
box_edge_len = 4.0
colors = ['#E24A33', '#348ABD', '#988ED5']
n_samples = 100

print("Generating start points...")
start_points = []
for i, center in enumerate(box_centers):
    pts = np.random.uniform(
        low = center - box_edge_len/2,
        high= center + box_edge_len/2,
        size=(n_samples, 2)
    )
    start_points.append(pts)
start_points = np.vstack(start_points)

print("Integrating ODEs backwards (Target t=1 -> Base t=0)...")
# We integrate from t=1 to t=0. 
t_eval = np.linspace(1, 0, 100)
trajectories = []

for pt in start_points:
    # solve_ivp requires t_span=(start, end)
    sol = solve_ivp(ode_func, [1, 0], pt, t_eval=t_eval, rtol=1e-5, atol=1e-6)
    if sol.success:
        trajectories.append(sol.y.T) # Store as (Time, 2)
    else:
        print(f"Solver failed for point {pt}")

print(f"Computed {len(trajectories)} trajectories.")

# ==========================================
# 3. Visualization
# ==========================================
fig, ax = plt.subplots()

# --- Background Densities ---
x_grid = np.linspace(-7, 7, 150)
y_grid = np.linspace(-7, 7, 150)
X, Y = np.meshgrid(x_grid, y_grid)
pos = np.dstack((X, Y))

# Target Density
Z_target = np.zeros_like(X)
for i in range(n_components):
    Z_target += weights[i] * multivariate_normal.pdf(pos, mean=means[i], cov=covs[i])
ax.contour(X, Y, Z_target, levels=5, colors='k', alpha=0.3, linestyles='-')

# Base Density
Z_base = multivariate_normal.pdf(pos, mean=[0,0], cov=np.eye(2))
ax.contour(X, Y, Z_base, levels=3, colors='k', alpha=0.3, linestyles='--')

# --- Trajectories ---
legend_proxies = []

# Loop through groups (assuming order matches box_centers)
for i in range(len(box_centers)):
    color = colors[i]
    
    # Target Box
    rect = patches.Rectangle(
        box_centers[i] - box_edge_len/2, box_edge_len, box_edge_len,
        linewidth=2, edgecolor=color, facecolor='none', zorder=20
    )
    ax.add_patch(rect)
    legend_proxies.append(rect)
    
    # Trajectories
    group_paths = trajectories[i*n_samples : (i+1)*n_samples]
    for path in group_paths:
        # Path is from t=1 down to t=0
        ax.plot(path[:, 0], path[:, 1], color=color, alpha=0.6, linewidth=1.5)
        # Start (Target, t=1)
        ax.scatter(path[0, 0], path[0, 1], color=color, s=20, zorder=10)
        # End (Base, t=0)
        ax.scatter(path[-1, 0], path[-1, 1], color=color, marker='x', s=40, alpha=0.8, zorder=10)

ax.set_xlim(-7, 7)
ax.set_ylim(-7, 7)
ax.set_aspect('equal')
ax.set_title("Flow Matching: Marginal Vector Field Integration")
ax.set_xlabel(r"$\theta_1$")
ax.set_ylabel(r"$\theta_2$")

# Custom Legend
legend_proxies.append(plt.Line2D([0], [0], color='gray', lw=1.5, label='Flow'))
legend_proxies.append(plt.Line2D([0], [0], marker='o', color='gray', lw=0, label=r'Target ($t=1$)'))
legend_proxies.append(plt.Line2D([0], [0], marker='x', color='gray', lw=0, label=r'Base ($t=0$)'))
ax.legend(handles=legend_proxies, loc='upper left')

plt.tight_layout()
filename = "flow_matching_curved.pdf"
plt.savefig(filename)
print(f"Plot saved to {filename}")
plt.show() # Commented out to prevent MacOS backend hang