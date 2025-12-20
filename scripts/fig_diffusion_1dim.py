import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import matplotlib.gridspec as gridspec

np.random.seed(42)

# --- Configuration ---
TARGETS = np.array([-7.0, 3.0, 7.0]) 
T_MAX = 9.0 
L_TRAJECTORIES = 32
TIME_STEPS = 2000

# Base Colors (RGB)
C_LEFT    = np.array(to_rgb("#ff7f0e")) # Orange
C_MID     = np.array(to_rgb("#2ca02c")) # Green
C_RIGHT   = np.array(to_rgb("#1f77b4")) # Blue
C_NEUTRAL = np.array(to_rgb("#d9d9d9")) # Light Gray

TARGET_COLORS = [C_LEFT, C_MID, C_RIGHT]
TARGET_NAMES = ["Mode A", "Mode B", "Mode C"]

# --- Physics ---
def alpha(t): return np.exp(-t)
def get_mean_var(x0, t): return x0 * np.sqrt(alpha(t)), 1.0 - alpha(t)

def get_atomic_drift(x, t, target):
    var = 1.0 - alpha(t)
    var = np.maximum(var, 1e-4)
    mean_atomic = target * np.sqrt(alpha(t))
    score_atomic = -(x - mean_atomic) / var
    drift = -0.5*x - score_atomic
    return drift

def get_weights_and_drift(x, t, targets):
    _, var = get_mean_var(0, t)
    var = np.maximum(var, 1e-4)
    
    # Weights
    mus = targets[:, np.newaxis] * np.sqrt(alpha(t))
    lps = -0.5 * ((x - mus)**2) / var
    max_lp = np.max(lps, axis=0)
    exps = np.exp(lps - max_lp)
    weights = exps / np.sum(exps, axis=0)
    
    # Drift
    means_atomic = targets[:, np.newaxis] * np.sqrt(alpha(t))
    scores_atomic = -(x - means_atomic) / var
    score_mix = np.sum(weights * scores_atomic, axis=0)
    drift = -0.5*x - score_mix
    
    return drift, weights

def get_arrow_color(weights):
    mix = (weights[0] * C_LEFT + 
           weights[1] * C_MID + 
           weights[2] * C_RIGHT)
    
    w_max = np.max(weights)
    confidence = (w_max - (1/3)) / (2/3)
    confidence = np.clip(confidence, 0, 1)
    confidence = confidence ** 2 
    
    final_color = confidence * mix + (1 - confidence) * C_NEUTRAL
    return final_color

# --- Simulation ---
def simulate(targets_to_use):
    dt = T_MAX / TIME_STEPS
    x = np.random.randn(L_TRAJECTORIES)
    trajs = [x.copy()]
    ts = np.linspace(T_MAX, 0, TIME_STEPS)
    
    for t in ts:
        if t < 1e-3: break
        
        if len(targets_to_use) == 1:
            d = get_atomic_drift(x, t, targets_to_use[0])
        else:
            d, _ = get_weights_and_drift(x, t, targets_to_use)
            
        x = x - d*dt + np.random.randn(L_TRAJECTORIES)*np.sqrt(dt)
        trajs.append(x.copy())
    return np.array(trajs), ts

# --- Plotting Helper ---
def plot_panel(ax, targets_to_use, panel_color, title, show_y_label=False):
    # 1. Generate Vector Field
    t_grid = np.linspace(0, T_MAX, 15)
    x_grid = np.linspace(-9, 9, 20)
    T_mesh, X_mesh = np.meshgrid(t_grid, x_grid)

    U, V = [], []
    Colors = []

    for i in range(T_mesh.shape[0]):
        for j in range(T_mesh.shape[1]):
            tau = T_mesh[i,j]
            t_phys = max(T_MAX - tau, 0.01)
            x = X_mesh[i,j]
            
            if len(targets_to_use) == 1:
                drift = get_atomic_drift(np.array([x]), t_phys, targets_to_use[0])
                rgb = panel_color
            else:
                drift, w_vec = get_weights_and_drift(np.array([x]), t_phys, targets_to_use)
                rgb = get_arrow_color(w_vec[:, 0])

            U.append(1.0)
            V.append(-drift[0])
            Colors.append(rgb)

    U = np.array(U).reshape(T_mesh.shape)*3
    V = np.array(V).reshape(T_mesh.shape)
    
    if len(targets_to_use) == 1:
        Colors = panel_color 
    else:
        Colors = np.array(Colors).reshape(T_mesh.shape[0]*T_mesh.shape[1], 3)

    # Normalize Arrows
    N = np.sqrt(U**2 + V**2)
    U, V = U/N, V/N

    # 2. Plot Quiver
    ax.quiver(T_mesh, X_mesh, U, V, color=Colors,
               scale=15, width=0.006, headwidth=3, alpha=1.0, zorder=10)

    # 3. Trajectories (All Black)
    trajs, ts = simulate(targets_to_use)
    plot_times = np.linspace(0, T_MAX, len(trajs))
    
    for i in range(L_TRAJECTORIES):
        # Using black with transparency for density effect
        ax.plot(plot_times, trajs[:, i], color='black', alpha=0.60, linewidth=0.8)

    ax.text(0.05, 0.95, title, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none'),
            zorder=30)
    ax.set_xlim(0, T_MAX)
    ax.set_ylim(-9, 9)
    ax.set_xlabel("Time $t$", fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=8)
    
    if show_y_label:
        ax.set_ylabel(r"Parameter $\theta$", fontsize=8)
    else:
        ax.set_yticklabels([])
    
    # Add Target Marker
    for t_val in targets_to_use:
        ax.scatter([T_MAX], [t_val], color='black', s=30, zorder=20, marker='x')

# --- Main Plot Setup ---
# A4 width is ~8.27 in. Margins reduce this. 6.2 is a safe content width.
fig = plt.figure(figsize=(6.3, 2.2))

# Snug fit: reduce wspace
gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.05,
                       left=0.08, right=0.99, bottom=0.22, top=0.95)

# Panel 1: Left Mode
ax1 = fig.add_subplot(gs[0, 0])
plot_panel(ax1, np.array([TARGETS[0]]), C_LEFT, TARGET_NAMES[0], show_y_label=True)

# Panel 2: Middle Mode
ax2 = fig.add_subplot(gs[0, 1])
plot_panel(ax2, np.array([TARGETS[1]]), C_MID, TARGET_NAMES[1])

# Panel 3: Right Mode
ax3 = fig.add_subplot(gs[0, 2])
plot_panel(ax3, np.array([TARGETS[2]]), C_RIGHT, TARGET_NAMES[2])

# Panel 4: Combined Mixture
ax4 = fig.add_subplot(gs[0, 3])
plot_panel(ax4, TARGETS, C_NEUTRAL, "Mixture")

#plt.suptitle("Additive Diffusion: The Global Field is the Sum of Atomic Fields", fontsize=12, y=0.98)
plt.savefig('figures/fig_diffusion_1dim.pdf', format='pdf', bbox_inches='tight')
#plt.show()