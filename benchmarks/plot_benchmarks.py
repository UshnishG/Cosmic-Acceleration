"""
Benchmark Visualization Plotter.
Plots runtime scaling, speedup curves, and error analysis across particle counts.
"""
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_benchmark_charts(
    results_path: str = "output/benchmark_results.json",
    output_chart_path: str = "output/benchmark_performance.png"
):
    if not os.path.exists(results_path):
        print(f"File {results_path} not found.")
        return
        
    with open(results_path, "r") as f:
        data = json.load(f)
        
    benchmarks = data["benchmarks"]
    N_vals = [b["N"] for b in benchmarks]
    
    cpu_times = [b["timings_ms"]["CPU Direct"] for b in benchmarks]
    gpu_dir_times = [b["timings_ms"]["GPU Direct"] for b in benchmarks]
    gpu_tiled_times = [b["timings_ms"]["GPU Tiled"] for b in benchmarks]
    bh_times = [b["timings_ms"]["Barnes-Hut (theta=0.5)"] for b in benchmarks]
    
    speedup_dir = [b["speedups"]["GPU Direct"] for b in benchmarks]
    speedup_tiled = [b["speedups"]["GPU Tiled"] for b in benchmarks]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor='white')
    
    # 1. Runtime vs N (log-log)
    ax1.plot(N_vals, cpu_times, 'o-', color='#d9534f', linewidth=2, label='CPU Direct (O(N²))')
    ax1.plot(N_vals, bh_times, 's--', color='#f0ad4e', linewidth=2, label='Barnes-Hut (O(N log N))')
    ax1.plot(N_vals, gpu_dir_times, '^-', color='#0275d8', linewidth=2, label='GPU Direct (CUDA)')
    ax1.plot(N_vals, gpu_tiled_times, 'd-', color='#5cb85c', linewidth=2, label='GPU Tiled (Shared Mem)')
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Number of Bodies (N)', fontsize=12)
    ax1.set_ylabel('Execution Time per Step (ms)', fontsize=12)
    ax1.set_title('Computational Scaling (Log-Log)', fontsize=14, fontweight='bold')
    ax1.grid(True, which='both', linestyle=':', alpha=0.6)
    ax1.legend(fontsize=10)
    
    # 2. Speedup vs CPU
    ax2.plot(N_vals, speedup_dir, '^-', color='#0275d8', linewidth=2.5, markersize=8, label='GPU Direct Speedup')
    ax2.plot(N_vals, speedup_tiled, 'd-', color='#5cb85c', linewidth=2.5, markersize=8, label='GPU Tiled Speedup')
    ax2.axhline(50.0, color='crimson', linestyle='--', linewidth=1.5, label='Poster Reported ~50× Target')
    
    ax2.set_xlabel('Number of Bodies (N)', fontsize=12)
    ax2.set_ylabel('Speedup Factor (× over CPU Direct)', fontsize=12)
    ax2.set_title(f'GPU Acceleration Factor ({data["hardware"]})', fontsize=14, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_chart_path), exist_ok=True)
    plt.savefig(output_chart_path, dpi=300)
    plt.close(fig)
    print(f"Saved benchmark performance chart to: {output_chart_path}")

if __name__ == "__main__":
    generate_benchmark_charts()
