"""
Cosmic Acceleration: Live Interactive N-Body Simulation.
Provides real-time interactive simulation viewers for live demonstrations and faculty defenses:
1. 3D Interactive Viewer (Matplotlib TkAgg): Full 3D orbital space with free mouse rotation,
   orbital trails, auto-camera rotation, and live HUD telemetry.
2. Fast 2D/3D Canvas Viewer (OpenCV): High-performance 60+ FPS glowing galaxy rendering
   with exponential alpha-decay trails and live telemetry.
3. Live Runtime Solver Switching: Seamlessly toggle between GPU Tiled, GPU Direct,
   CPU NumPy, and Barnes-Hut with a single keypress to show real-time performance differences!
"""
import os
import sys
import time
import argparse
import numpy as np

from src.constants import SimulationConfig
from src.initial_conditions import generate_keplerian_disk
from src.cpu_solver import CPUNBodySolver
from src.gpu_direct_solver import GPUNBodyDirectSolver, get_available_device
from src.gpu_tiled_solver import GPUTiledNBodySolver
from src.barnes_hut import BarnesHutSolver
from src.integrators import compute_system_metrics, compute_adaptive_dt

SOLVER_CHOICES = ["gpu_tiled", "gpu_direct", "cpu", "barnes_hut"]
SOLVER_NAMES = {
    "gpu_tiled": "GPU CUDA Tiled (sm_89 Shared Memory)",
    "gpu_direct": "GPU CUDA Direct (Global Memory)",
    "cpu": "CPU Vectorized (NumPy Baseline)",
    "barnes_hut": "Barnes-Hut Octree (O(N log N))"
}

class LiveSimulationEngine:
    """Manages the simulation state, solver transitions, and performance metrics."""
    def __init__(
        self,
        config: SimulationConfig,
        solver_name: str = "gpu_tiled",
        is_3d: bool = True,
        adaptive: bool = False,
        seed: int = 42
    ):
        self.config = config
        self.is_3d = is_3d
        self.adaptive = adaptive
        self.seed = seed
        self.paused = False
        self.substeps = 1
        self.solver_name = solver_name
        self.solver_idx = SOLVER_CHOICES.index(solver_name) if solver_name in SOLVER_CHOICES else 0
        
        # Performance tracking
        self.frame_times = []
        self.physics_step_times = []
        self.last_frame_timestamp = time.perf_counter()
        
        # Reset to initial state
        self.reset()

    def reset(self):
        """Re-initializes particles to stable inclined Keplerian disk."""
        self.pos, self.vel, self.mass = generate_keplerian_disk(
            self.config, is_3d=self.is_3d, seed=self.seed
        )
        self.N = len(self.mass)
        self.sim_time = 0.0
        self.step_count = 0
        
        # Initial energy baseline
        m0 = compute_system_metrics(
            self.pos, self.vel, self.mass, self.config.G, self.config.softening
        )
        self.initial_energy = m0["total_energy"]
        self.current_metrics = m0
        
        # Initialize solver
        self._init_solver(self.solver_name)

    def _init_solver(self, name: str):
        self.solver_name = name
        if name == "gpu_tiled":
            self.solver = GPUTiledNBodySolver(self.pos, self.vel, self.mass, self.config)
        elif name == "gpu_direct":
            self.solver = GPUNBodyDirectSolver(self.pos, self.vel, self.mass, self.config)
        elif name == "cpu":
            self.solver = CPUNBodySolver(self.pos, self.vel, self.mass, self.config)
        elif name == "barnes_hut":
            self.solver = BarnesHutSolver(self.pos, self.vel, self.mass, self.config)
        else:
            self.solver = GPUTiledNBodySolver(self.pos, self.vel, self.mass, self.config)

    def cycle_solver(self):
        """Hot-swaps to the next solver preserving identical physical positions and velocities."""
        pos, vel, acc, mass = self.solver.get_state()
        self.pos, self.vel, self.mass = pos, vel, mass
        
        self.solver_idx = (self.solver_idx + 1) % len(SOLVER_CHOICES)
        new_name = SOLVER_CHOICES[self.solver_idx]
        print(f"\n[HOT-SWAP] Switching Solver -> {SOLVER_NAMES[new_name]}")
        self._init_solver(new_name)
        return self.solver_name

    def step_simulation(self):
        """Advances the simulation by one frame (executing self.substeps physics steps)."""
        if self.paused:
            return
            
        t0 = time.perf_counter()
        for _ in range(self.substeps):
            if self.adaptive:
                dt = compute_adaptive_dt(
                    self.solver.acc if hasattr(self.solver, 'acc') else self.solver.get_state()[2],
                    self.config.softening,
                    eta=self.config.eta_adaptive,
                    dt_min=self.config.dt_min,
                    dt_max=self.config.dt_max
                )
            else:
                dt = self.config.dt_fixed
                
            self.solver.step(dt)
            self.sim_time += dt
            self.step_count += 1
            
        t1 = time.perf_counter()
        step_elapsed = (t1 - t0) / self.substeps
        self.physics_step_times.append(step_elapsed)
        if len(self.physics_step_times) > 60:
            self.physics_step_times.pop(0)

        # Update cached state
        self.pos, self.vel, _, self.mass = self.solver.get_state()

        # Update metrics every 5 steps to save compute
        if self.step_count % 5 == 0:
            self.current_metrics = compute_system_metrics(
                self.pos, self.vel, self.mass, self.config.G, self.config.softening
            )

        # Track render FPS
        now = time.perf_counter()
        dt_frame = now - self.last_frame_timestamp
        self.last_frame_timestamp = now
        if dt_frame > 0:
            self.frame_times.append(dt_frame)
            if len(self.frame_times) > 30:
                self.frame_times.pop(0)

    @property
    def fps(self) -> float:
        if not self.frame_times:
            return 0.0
        return len(self.frame_times) / sum(self.frame_times)

    @property
    def mean_step_ms(self) -> float:
        if not self.physics_step_times:
            return 0.0
        return (sum(self.physics_step_times) / len(self.physics_step_times)) * 1000.0

    @property
    def energy_drift_pct(self) -> float:
        if abs(self.initial_energy) < 1e-12:
            return 0.0
        return abs((self.current_metrics["total_energy"] - self.initial_energy) / self.initial_energy) * 100.0


# =============================================================================
# GUI 1: INTERACTIVE 3D MATPLOTLIB VIEWER (TkAgg)
# =============================================================================
def run_live_3d(engine: LiveSimulationEngine, trail_len: int = 40):
    """
    Launches interactive 3D Matplotlib window with mouse rotation/zoom,
    orbital fading trails, and a real-time Heads-Up Display (HUD).
    """
    import matplotlib
    # Use TkAgg for real-time GUI on Windows
    try:
        matplotlib.use('TkAgg')
    except Exception:
        pass
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(13, 9), facecolor='#0B0F19')
    fig.canvas.manager.set_window_title("Cosmic Acceleration: Interactive 3D CUDA Simulation")
    ax = fig.add_subplot(111, projection='3d', facecolor='#0B0F19')

    # Styling panes
    pane_col = (0.12, 0.16, 0.24, 0.9)
    ax.xaxis.set_pane_color(pane_col)
    ax.yaxis.set_pane_color(pane_col)
    ax.zaxis.set_pane_color(pane_col)
    ax.tick_params(colors='#94A3B8', labelsize=9)
    ax.set_xlabel('X', color='#94A3B8', fontsize=11, labelpad=8)
    ax.set_ylabel('Y', color='#94A3B8', fontsize=11, labelpad=8)
    ax.set_zlabel('Z', color='#94A3B8', fontsize=11, labelpad=8)

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_zlim(-0.12, 0.12)
    ax.view_init(elev=28, azim=45)

    # Plasma colors for orbiters
    N = engine.N
    colors = plt.cm.plasma(np.linspace(0.15, 0.95, N - 1))

    # Trail history buffer: deque of (N, 3)
    trails = [engine.pos.copy()]
    trail_lines = []
    for i in range(1, N):
        line, = ax.plot([], [], [], color=colors[i - 1], linewidth=0.8, alpha=0.65)
        trail_lines.append(line)

    # Scatter points for orbiters and central star
    scatter_orbiters = ax.scatter(
        engine.pos[1:, 0], engine.pos[1:, 1], engine.pos[1:, 2],
        c=colors, s=14, depthshade=False, zorder=50
    )
    scatter_center = ax.scatter(
        [engine.pos[0, 0]], [engine.pos[0, 1]], [engine.pos[0, 2]],
        c='gold', edgecolors='white', s=260, depthshade=False, zorder=100
    )

    # HUD Text elements (Screen 2D coordinates)
    hud_title = fig.text(
        0.02, 0.96, "COSMIC ACCELERATION: 3D N-BODY SIMULATION",
        color='#FFFFFF', fontsize=13, fontweight='bold', family='sans-serif'
    )
    hud_solver = fig.text(
        0.02, 0.93, f"Solver: {SOLVER_NAMES[engine.solver_name]}",
        color='#38BDF8', fontsize=10.5, fontweight='bold', family='sans-serif'
    )
    hud_telemetry = fig.text(
        0.02, 0.895, "",
        color='#E2E8F0', fontsize=9.5, family='monospace'
    )
    hud_energy = fig.text(
        0.02, 0.865, "",
        color='#34D399', fontsize=9.5, family='monospace'
    )
    hud_controls = fig.text(
        0.02, 0.02,
        "[Space] Pause | [C] Cycle Solver | [R] Reset | [A] Adaptive Δt | [T] Trails | [V] Rotate | [+/-] Speed | [Q] Quit",
        color='#94A3B8', fontsize=9, family='sans-serif'
    )

    # State flags
    show_trails = True
    auto_rotate = True
    is_running = [True]

    def on_key(event):
        nonlocal show_trails, auto_rotate
        if event.key in ['q', 'escape']:
            is_running[0] = False
            plt.close(fig)
        elif event.key == ' ':
            engine.paused = not engine.paused
        elif event.key in ['c', 'C']:
            engine.cycle_solver()
            hud_solver.set_text(f"Solver: {SOLVER_NAMES[engine.solver_name]}")
        elif event.key in ['r', 'R']:
            engine.reset()
            trails.clear()
            trails.append(engine.pos.copy())
        elif event.key in ['a', 'A']:
            engine.adaptive = not engine.adaptive
        elif event.key in ['t', 'T']:
            show_trails = not show_trails
            if not show_trails:
                for line in trail_lines:
                    line.set_data([], [])
                    line.set_3d_properties([])
        elif event.key in ['v', 'V']:
            auto_rotate = not auto_rotate
        elif event.key in ['+', '=']:
            engine.substeps = min(engine.substeps * 2, 16)
        elif event.key in ['-', '_']:
            engine.substeps = max(engine.substeps // 2, 1)

    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.tight_layout()
    plt.show(block=False)

    print("\n=======================================================")
    print("LIVE 3D SIMULATION RUNNING")
    print("Controls:")
    print("  [Space]   : Pause / Play")
    print("  [C]       : Cycle Solver (GPU Tiled -> Direct -> CPU -> Barnes-Hut)")
    print("  [R]       : Reset to t = 0")
    print("  [A]       : Toggle Adaptive Time-Stepping")
    print("  [T]       : Toggle Trails")
    print("  [V]       : Toggle Camera Auto-Rotation")
    print("  [+] / [-] : Speed Up / Slow Down")
    print("  [Q] / Esc : Quit")
    print("=======================================================\n")

    current_azim = 45.0
    while is_running[0] and plt.fignum_exists(fig.number):
        engine.step_simulation()

        # Update central star & orbiters
        scatter_center._offsets3d = ([engine.pos[0, 0]], [engine.pos[0, 1]], [engine.pos[0, 2]])
        scatter_orbiters._offsets3d = (engine.pos[1:, 0], engine.pos[1:, 1], engine.pos[1:, 2])

        # Update trails
        if show_trails and not engine.paused:
            trails.append(engine.pos.copy())
            if len(trails) > trail_len:
                trails.pop(0)

            trail_arr = np.array(trails)  # (K, N, 3)
            for i in range(1, N):
                trail_lines[i - 1].set_data(trail_arr[:, i, 0], trail_arr[:, i, 1])
                trail_lines[i - 1].set_3d_properties(trail_arr[:, i, 2])

        # Auto-rotate camera
        if auto_rotate and not engine.paused:
            current_azim = (current_azim + 0.35) % 360.0
            ax.view_init(elev=28, azim=current_azim)

        # Update HUD text
        status_str = "PAUSED" if engine.paused else "RUNNING"
        telemetry_str = (
            f"Status: {status_str:<7} | Step: {engine.step_count:<5} | Time: {engine.sim_time:6.2f}s | "
            f"FPS: {engine.fps:4.1f} | Latency: {engine.mean_step_ms:5.2f} ms/step"
        )
        hud_telemetry.set_text(telemetry_str)

        adaptive_str = "ON" if engine.adaptive else "OFF"
        energy_str = (
            f"Energy Drift: |ΔE/E₀| = {engine.energy_drift_pct:.4f}% | "
            f"Substeps: {engine.substeps}x | Adaptive Δt: {adaptive_str}"
        )
        hud_energy.set_text(energy_str)

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        time.sleep(0.001)


# =============================================================================
# GUI 2: HIGH-FPS OPENCV GLOWING CANVAS (2D / 3D Projection)
# =============================================================================
def run_live_opencv(engine: LiveSimulationEngine):
    """
    High-performance 60-120 FPS hardware-accelerated OpenCV renderer.
    Features glowing blooming star, neon plasma particles, alpha fading trails,
    and a real-time HUD with telemetry.
    """
    import cv2
    import matplotlib.cm as cm

    width, height = 1000, 1000
    scale = 360.0
    cx, cy = width // 2, height // 2

    # Persistent trail canvas for exponential alpha decay
    trail_canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # Pre-generate plasma colors in BGR
    N = engine.N
    plasma_cmap = cm.plasma(np.linspace(0.15, 0.95, N - 1))
    colors_bgr = []
    for rgba in plasma_cmap:
        b = int(rgba[2] * 255)
        g = int(rgba[1] * 255)
        r = int(rgba[0] * 255)
        colors_bgr.append((b, g, r))

    window_name = "Cosmic Acceleration: High-FPS GPU Simulation"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("\n=======================================================")
    print("LIVE HIGH-FPS OPENCV SIMULATION RUNNING")
    print("Hotkeys in window:")
    print("  [Space]   : Pause / Play")
    print("  [C]       : Cycle Solver (GPU Tiled -> Direct -> CPU -> Barnes-Hut)")
    print("  [R]       : Reset")
    print("  [A]       : Toggle Adaptive Δt")
    print("  [+] / [-] : Speed Up / Slow Down")
    print("  [Q] / Esc : Quit")
    print("=======================================================\n")

    while True:
        engine.step_simulation()

        # Fade previous trails (exponential decay: 92% retention per frame)
        trail_canvas = cv2.multiply(trail_canvas, np.full_like(trail_canvas, 0.93))

        # Project particles to 2D screen coordinates
        coords_x = (cx + engine.pos[:, 0] * scale).astype(int)
        coords_y = (cy - engine.pos[:, 1] * scale).astype(int)

        # Draw orbiters on trail canvas
        for i in range(1, N):
            px, py = coords_x[i], coords_y[i]
            if 0 <= px < width and 0 <= py < height:
                col = colors_bgr[i - 1]
                cv2.circle(trail_canvas, (px, py), 2, col, -1, lineType=cv2.LINE_AA)

        # Compose frame
        frame = trail_canvas.copy()

        # Draw glowing central star (Body 0)
        c_px, c_py = coords_x[0], coords_y[0]
        if 0 <= c_px < width and 0 <= c_py < height:
            # Multi-layer radial halo
            cv2.circle(frame, (c_px, c_py), 24, (0, 70, 160), -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (c_px, c_py), 15, (0, 140, 240), -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (c_px, c_py), 8, (0, 220, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (c_px, c_py), 4, (255, 255, 255), -1, lineType=cv2.LINE_AA)

        # Draw current particle heads with glowing dots
        for i in range(1, N):
            px, py = coords_x[i], coords_y[i]
            if 0 <= px < width and 0 <= py < height:
                col = colors_bgr[i - 1]
                cv2.circle(frame, (px, py), 3, col, -1, lineType=cv2.LINE_AA)
                cv2.circle(frame, (px, py), 1, (255, 255, 255), -1, lineType=cv2.LINE_AA)

        # Draw semi-transparent HUD banner at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 95), (15, 20, 30), -1)
        cv2.rectangle(overlay, (0, height - 35), (width, height), (15, 20, 30), -1)
        frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

        # Text Telemetry
        cv2.putText(frame, "COSMIC ACCELERATION: HIGH-FPS N-BODY ENGINE", 
                    (20, 28), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
        
        solver_str = f"Solver: {SOLVER_NAMES[engine.solver_name]}"
        cv2.putText(frame, solver_str, (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 180, 50), 1, cv2.LINE_AA)

        status_str = "PAUSED" if engine.paused else "RUNNING"
        telemetry = (
            f"Status: {status_str} | Step: {engine.step_count} | Sim Time: {engine.sim_time:.2f}s | "
            f"FPS: {engine.fps:.1f} | Latency: {engine.mean_step_ms:.2f} ms | Drift: {engine.energy_drift_pct:.4f}%"
        )
        cv2.putText(frame, telemetry, (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 220, 240), 1, cv2.LINE_AA)

        controls_str = "[Space] Pause | [C] Cycle Solver | [R] Reset | [A] Adaptive Δt | [+/-] Speed | [Q] Quit"
        cv2.putText(frame, controls_str, (20, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 170, 180), 1, cv2.LINE_AA)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), ord('Q'), 27]:  # 27 is ESC
            break
        elif key == ord(' '):
            engine.paused = not engine.paused
        elif key in [ord('c'), ord('C')]:
            engine.cycle_solver()
        elif key in [ord('r'), ord('R')]:
            engine.reset()
            trail_canvas = np.zeros((height, width, 3), dtype=np.uint8)
        elif key in [ord('a'), ord('A')]:
            engine.adaptive = not engine.adaptive
        elif key in [ord('+'), ord('=')]:
            engine.substeps = min(engine.substeps * 2, 16)
        elif key in [ord('-'), ord('_')]:
            engine.substeps = max(engine.substeps // 2, 1)

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Cosmic Acceleration: Live Interactive N-Body Simulation")
    parser.add_argument("--gui", choices=["3d", "2d", "fast"], default="3d",
                        help="GUI frontend: '3d' (Matplotlib TkAgg with 3D rotation) or '2d'/'fast' (OpenCV high FPS)")
    parser.add_argument("--solver", choices=SOLVER_CHOICES, default="gpu_tiled",
                        help="Initial acceleration compute engine")
    parser.add_argument("--particles", type=int, default=150, help="Number of bodies (default: 150)")
    parser.add_argument("--adaptive", action="store_true", help="Start with adaptive time-stepping enabled")
    parser.add_argument("--substeps", type=int, default=1, help="Physics steps per render frame")
    args = parser.parse_args()

    config = SimulationConfig(num_bodies=args.particles)
    engine = LiveSimulationEngine(
        config=config,
        solver_name=args.solver,
        is_3d=(args.gui == "3d"),
        adaptive=args.adaptive
    )
    engine.substeps = args.substeps

    if args.gui == "3d":
        run_live_3d(engine)
    else:
        run_live_opencv(engine)


if __name__ == "__main__":
    main()
