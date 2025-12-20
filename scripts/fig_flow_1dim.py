import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import matplotlib.gridspec as gridspec

np.random.seed(42)

# --- Configuration ---
TARGETS = np.array([-7.0, 3.0, 7.0])
T_MAX = 0.99  # Flow Matching standard is t in [0, 1]
L_TRAJECTORIES = 32
TIME_STEPS = 2000

# Base Colors (RGB)
C_LEFT    = np.array(to_rgb("#ff7f0e")) # Orange
C_MID     = np.array(to_rgb("#2ca02c")) # Green
C_RIGHT   = np.array(to_rgb("#1f77b4")) # Blue
C_NEUTRAL = np.array(to_rgb("#d9d9d9")) # Light Gray

TARGET_COLORS = [C_LEFT, C_MID, C_RIGHT]
TARGET_NAMES = ["Mode A", "Mode B", "Mode C"]

# --- Physics: Optimal Transport Flow Matching ---
# Path: x_t = (1-t)*x_0 + t*x_1
# If x_0 ~ N(0,1), then x_t ~ N(t*x_1, (1-t)^2)

def get_mean_std(target, t):
    # Mean moves linearly from 0 to target
    mean = t * target
    # Std shrinks linearly from 1 to 0
    std = (1 - t)
    return mean, std

def get_atomic_velocity(x, t, target):
    """Velocity for a single target."""
    t = np.minimum(t, 0.999)  # Singularity guard
    # v_t(x|x_1) = (x_1 - x) / (1 - t)
    velocity = (target - x) / (1 - t)
    return velocity

def get_weights_and_velocity(x, t, targets):
    """Velocity for a mixture of targets."""
    t = np.minimum(t, 0.999)  # Singularity guard

    # 1. Weights (Posterior Probability)
    # We compare densities of N(t*x_i, (1-t)^2)
    means = t * targets[:, np.newaxis]
    var = (1 - t)**2

    # Log-likelihoods
    lps = -0.5 * ((x - means)**2) / var
    max_lp = np.max(lps, axis=0)
    exps = np.exp(lps - max_lp)
    weights = exps / np.sum(exps, axis=0)  # Shape: (N_targets, N_x)

    # 2. Conditional Velocities
    # v_t(x|x_1) = (x_1 - x) / (1 - t)
    targets_broad = targets[:, np.newaxis]
    velocities_atomic = (targets_broad - x) / (1 - t)

    # 3. Marginal Velocity (Expected Velocity)
    velocity_mix = np.sum(weights * velocities_atomic, axis=0)

    return velocity_mix, weights

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

# --- Simulation (ODE Integration - Deterministic) ---
def simulate(targets_to_use):
    dt = T_MAX / TIME_STEPS
    x = np.random.randn(L_TRAJECTORIES)
    trajs = [x.copy()]
    ts = np.linspace(0, T_MAX, TIME_STEPS)

    for t in ts:
        if t >= 0.995: break  # Stop just before singularity

        if len(targets_to_use) == 1:
            v = get_atomic_velocity(x, t, targets_to_use[0])
        else:
            v, _ = get_weights_and_velocity(x, t, targets_to_use)

        # Euler Update: x_new = x + v * dt (Deterministic ODE!)
        x = x + v * dt
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
            t_val = T_mesh[i,j]
            x = X_mesh[i,j]

            if len(targets_to_use) == 1:
                velocity = get_atomic_velocity(np.array([x]), t_val, targets_to_use[0])
                rgb = panel_color
            else:
                velocity, w_vec = get_weights_and_velocity(np.array([x]), t_val, targets_to_use)
                rgb = get_arrow_color(w_vec[:, 0])

            U.append(1.0)
            V.append(velocity[0])
            Colors.append(rgb)

    U = np.array(U).reshape(T_mesh.shape)*10
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
fig = plt.figure(figsize=(6.3, 2.2))

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

plt.savefig('figures/fig_flow_1dim.pdf', format='pdf', bbox_inches='tight')
#plt.show()
