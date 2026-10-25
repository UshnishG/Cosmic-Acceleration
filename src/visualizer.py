"""
Visualization Engine for Cosmic Acceleration N-Body Simulation.
Reproduces the exact poster visualizations consuming genuine simulation trajectory data:
1. 3D Physical Orbit Paths ("150-Body 3D Orbits (GPU Accelerated)") with dark background & plasma colormap
2. 3D Space-Time Trajectory Plot ("150-Body 3D Orbit Paths") where vertical axis represents Time
3. 2D Animated Fading Trail Orbit View
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from typing import Optional

def plot_3d_physical_orbits(
    trajectory: np.ndarray,
    output_path: str = "output/150_body_3d_orbits_gpu.png",
    title: str = "150-Body 3D Orbits (GPU Accelerated)",
    dpi: int = 300
):
    """
    Renders the exact 3D physical coordinate plot shown in the poster:
    Dark theme, grey coordinate panes, glowing yellow central mass, plasma-colored orbit trails.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # trajectory shape: (num_timesteps, N, 3)
    num_steps, N, _ = trajectory.shape
    
    fig = plt.figure(figsize=(10, 9), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    
    # Customize 3D pane colors to match poster grey panes
    pane_color = (0.35, 0.35, 0.35, 0.8)
    ax.xaxis.set_pane_color(pane_color)
    ax.yaxis.set_pane_color(pane_color)
    ax.zaxis.set_pane_color(pane_color)
    
    # White axis ticks and labels
    ax.tick_params(colors='white', labelsize=11)
    ax.set_xlabel('X', color='white', fontsize=12, labelpad=10)
    ax.set_ylabel('Y', color='white', fontsize=12, labelpad=10)
    ax.set_zlabel('Z', color='white', fontsize=12, labelpad=10)
    
    # Set title
    ax.set_title(title, color='white', fontsize=16, pad=20)
    
    # Color map for orbiters: plasma
    colors = plt.cm.plasma(np.linspace(0.15, 0.95, N - 1))
    
    # Plot orbit trails
    for i in range(1, N):
        x = trajectory[:, i, 0]
        y = trajectory[:, i, 1]
        z = trajectory[:, i, 2]
        ax.plot(x, y, z, color=colors[i - 1], linewidth=0.85, alpha=0.75)
        
    # Plot central heavy body at origin (Body 0)
    c_x = trajectory[-1, 0, 0]
    c_y = trajectory[-1, 0, 1]
    c_z = trajectory[-1, 0, 2]
    ax.scatter(
        [c_x], [c_y], [c_z],
        color='gold',
        edgecolors='white',
        s=220,
        depthshade=False,
        zorder=100
    )
    
    # Set aspect and view angle matching poster
    ax.view_init(elev=28, azim=45)
    
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_zlim(-0.065, 0.065)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, facecolor='black', edgecolor='none')
    plt.close(fig)
    print(f"Saved 3D physical orbits plot to: {output_path}")

def plot_3d_spacetime_paths(
    trajectory: np.ndarray,
    timestamps: np.ndarray,
    output_path: str = "output/150_body_3d_orbit_paths.png",
    title: str = "150-Body 3D Orbit Paths",
    dpi: int = 300,
    max_legend_bodies: int = 37
):
    """
    Renders the exact 3D Space-Time plot shown in the poster:
    White background, vertical Z-axis is Time, spiraling orbits, legend listing bodies.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    num_steps, N, _ = trajectory.shape
    
    fig = plt.figure(figsize=(11, 9), facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='white')
    
    # White background & clear grid lines
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.tick_params(colors='black', labelsize=11)
    ax.set_xlabel('X', color='black', fontsize=13, labelpad=10)
    ax.set_ylabel('Y', color='black', fontsize=13, labelpad=10)
    ax.set_zlabel('Time', color='black', fontsize=13, labelpad=10)
    ax.set_title(title, color='black', fontsize=16, pad=20)
    
    cmap = plt.cm.tab10
    
    # Plot space-time curves: (x(t), y(t), t)
    for i in range(1, N):
        x = trajectory[:, i, 0]
        y = trajectory[:, i, 1]
        t = timestamps
        color = cmap((i - 1) % 10)
        label = f"Body {i}" if i <= max_legend_bodies else None
        ax.plot(x, y, t, color=color, linewidth=1.1, alpha=0.85, label=label)
        
    # Central body space-time trajectory
    ax.plot(
        trajectory[:, 0, 0],
        trajectory[:, 0, 1],
        timestamps,
        color='gold',
        linewidth=2.5,
        alpha=0.9
    )
    # Scatter yellow dot at the top of the central body
    ax.scatter(
        [trajectory[-1, 0, 0]],
        [trajectory[-1, 0, 1]],
        [timestamps[-1]],
        color='gold',
        edgecolors='darkorange',
        s=180,
        zorder=100
    )
    
    ax.view_init(elev=25, azim=-55)
    
    # Legend on the left matching poster
    if max_legend_bodies > 0:
        ax.legend(
            loc='center left',
            bbox_to_anchor=(-0.15, 0.5),
            fontsize=8,
            frameon=True,
            ncol=1
        )
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, facecolor='white')
    plt.close(fig)
    print(f"Saved 3D space-time plot to: {output_path}")

def create_2d_fading_trails_animation(
    trajectory: np.ndarray,
    output_path: str = "output/2d_fading_trails.gif",
    trail_length: int = 25,
    fps: int = 30,
    stride: int = 1
):
    """
    Renders 2D animated view showing orbit trails fading over time (Poster Step 7).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    num_steps, N, _ = trajectory.shape
    sampled_indices = range(0, num_steps, stride)
    
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
    ax.set_facecolor('black')
    
    max_r = float(np.max(np.abs(trajectory[:, :, :2]))) * 1.15
    ax.set_xlim(-max_r, max_r)
    ax.set_ylim(-max_r, max_r)
    ax.set_aspect('equal')
    ax.axis('off')
    
    colors = plt.cm.plasma(np.linspace(0.2, 0.95, N - 1))
    
    # Central body marker
    central_dot, = ax.plot([], [], 'o', color='gold', markeredgecolor='white', markersize=12)
    # Orbiters current positions
    orbiters_dots, = ax.plot([], [], 'o', color='cyan', markersize=3, alpha=0.9)
    
    # Trail lines for each body
    trail_lines = [
        ax.plot([], [], '-', color=colors[i - 1], linewidth=1.0, alpha=0.7)[0]
        for i in range(1, N)
    ]
    
    def init():
        central_dot.set_data([], [])
        orbiters_dots.set_data([], [])
        for line in trail_lines:
            line.set_data([], [])
        return [central_dot, orbiters_dots] + trail_lines

    def update(frame_idx):
        step = sampled_indices[frame_idx]
        
        # Update central body
        central_dot.set_data([trajectory[step, 0, 0]], [trajectory[step, 0, 1]])
        
        # Update orbiters current points
        orbiters_dots.set_data(trajectory[step, 1:, 0], trajectory[step, 1:, 1])
        
        # Update fading trails
        start_step = max(0, step - trail_length)
        for i in range(1, N):
            tx = trajectory[start_step:step + 1, i, 0]
            ty = trajectory[start_step:step + 1, i, 1]
            trail_lines[i - 1].set_data(tx, ty)
            
        return [central_dot, orbiters_dots] + trail_lines

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(sampled_indices),
        interval=1000 // fps,
        blit=True
    )
    
    anim.save(output_path, writer='pillow', fps=fps)
    plt.close(fig)
    print(f"Saved 2D fading trails animation to: {output_path}")
