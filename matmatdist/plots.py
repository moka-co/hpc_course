#!/usr/bin/env python3
"""
Read matmatdist result files and generate matplotlib plots
Files named: matmatdist_result_N_NTROW_NTCOL_DB_PROW_PCOL.txt
"""

import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def parse_result_file(filepath):
    """Parse a single result file with 4 methods."""
    # Extract parameters from filename
    basename = os.path.basename(filepath)
    match = re.match(r'matmatdist_result_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)\.txt', basename)
    
    if not match:
        print(f"Warning: Could not parse filename {filepath}")
        return None
    
    N, NTROW, NTCOL, DB, PROW, PCOL = map(int, match.groups())
    
    data = {'N': N, 'NTROW': NTROW, 'NTCOL': NTCOL, 'DB': DB, 'PROW': PROW, 'PCOL': PCOL}
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Parse: method time=X gflops=X speedup=X efficiency=X
                parts = line.split()
                method = parts[0]
                
                for part in parts[1:]:
                    if '=' in part:
                        key, value = part.split('=')
                        data[f'{method}_{key}'] = float(value)
    
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None
    
    return data

def read_all_results(directory='./results'):
    """Read all result files from directory."""
    pattern = os.path.join(directory, 'matmatdist_result_*.txt')
    files = glob.glob(pattern)
    
    if not files:
        print(f"No result files found matching pattern: {pattern}")
        return None
    
    results = []
    for filepath in sorted(files):
        data = parse_result_file(filepath)
        if data:
            results.append(data)
    
    if not results:
        print("No valid results parsed")
        return None
    
    df = pd.DataFrame(results)
    return df

def plot_speedup(df, output_file='plots/speedup.png'):
    """Plot speedup for all 4 methods."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    n_values = sorted(df['N'].unique())
    methods = ['matmatblock', 'matmatthread', 'matmatdist']
    colors = ['#FF6B6B', '#4ECDC4', '#FFA62B']
    markers = ['s', 'o', 'D']
    
    x_pos = np.arange(len(n_values))
    width = 0.25
    
    for idx, method in enumerate(methods):
        speedups = [df[df['N'] == n][f'{method}_speedup'].values[0] for n in n_values]
        #ax.bar(x_pos + idx * width, speedups, width, label=method.replace('matmat', ''),color=colors[idx], alpha=0.8)
        ax.plot(n_values, speedups, marker=markers[idx], linewidth=2.5, markersize=8, label=method.replace('matmat', ''), color=colors[idx])
    
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Matrix Size (N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup', fontsize=12, fontweight='bold')
    ax.set_title('Speedup vs Baseline (matmat_ikj)', fontsize=14, fontweight='bold')
    ax.set_xticks(n_values)
    ax.set_xticklabels([f'{int(n)}' for n in n_values], fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_efficiency(df, output_file='plots/efficiency.png'):
    """Plot efficiency for all 4 methods."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    n_values = sorted(df['N'].unique())
    methods = ['matmatblock', 'matmatthread', 'matmatdist']
    colors = ['#FF6B6B', '#4ECDC4', '#FFA62B']
    markers = ['s', 'o', 'D']
    
    for idx, method in enumerate(methods):
        efficiencies = [df[df['N'] == n][f'{method}_efficiency'].values[0] for n in n_values]
        ax.plot(n_values, efficiencies, marker=markers[idx], linewidth=2.5, markersize=8, 
                label=method.replace('matmat', ''), color=colors[idx])
    
    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Perfect')
    ax.set_xlabel('Matrix Size (N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Efficiency', fontsize=12, fontweight='bold')
    ax.set_title('Parallel Efficiency', fontsize=14, fontweight='bold')
    ax.set_xticks(n_values)
    ax.set_xticklabels([f'{int(n)}' for n in n_values], fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_speedup_vs_processes(df, output_file='plots/speedup_vs_processes.png'):
    """Plot speedup vs number of processes for matmatdist."""
    n_values = sorted(df['N'].unique())
    fig, axes = plt.subplots(1, len(n_values), figsize=(5*len(n_values), 6))
    if len(n_values) == 1:
        axes = [axes]
    
    for ax_idx, n in enumerate(n_values):
        subset = df[df['N'] == n]
        
        # Calculate number of processes
        subset_copy = subset.copy()
        subset_copy['np'] = subset_copy['PROW'] * subset_copy['PCOL']
        
        ax = axes[ax_idx]
        
        methods = ['matmatdist']
        colors = ['#FFA62B']
        markers = ['o']
        
        for idx, method in enumerate(methods):
            # Sort by np for a clean line
            sub_sorted = subset_copy.sort_values('np')
            speedups = sub_sorted[f'{method}_speedup'].values
            np_vals = sub_sorted['np'].values
            ax.plot(np_vals, speedups, marker=markers[idx], linewidth=2.5, markersize=8,
                    label=method.replace('matmat', ''), color=colors[idx])
        
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Number of Processes (NP)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Speedup', fontsize=11, fontweight='bold')
        ax.set_title(f'N = {int(n)} (Distributed)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_speedup_vs_threads(df, output_file='plots/speedup_vs_threads.png'):
    """Plot speedup vs number of threads for matmatthread."""
    n_values = sorted(df['N'].unique())
    fig, axes = plt.subplots(1, len(n_values), figsize=(5*len(n_values), 6))
    if len(n_values) == 1:
        axes = [axes]
    
    for ax_idx, n in enumerate(n_values):
        subset = df[df['N'] == n]
        
        # Calculate total threads
        subset_copy = subset.copy()
        subset_copy['threads'] = subset_copy['NTROW'] * subset_copy['NTCOL']
        
        ax = axes[ax_idx]
        
        # Use matmatthread for this plot
        method = 'matmatthread'
        color = '#4ECDC4'
        marker = 's'
        
        # Sort by threads for a clean line
        sub_sorted = subset_copy.sort_values('threads')
        speedups = sub_sorted[f'{method}_speedup'].values
        thread_vals = sub_sorted['threads'].values
        
        ax.plot(thread_vals, speedups, marker=marker, linewidth=2.5, markersize=8,
                label='thread', color=color)
        
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Number of Threads', fontsize=11, fontweight='bold')
        ax.set_ylabel('Speedup', fontsize=11, fontweight='bold')
        ax.set_title(f'N = {int(n)} (Threaded)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_speedup_dist_thread_map(df, output_file='plots/speedup_dist_thread_map.png'):
    """Plot speedup vs NP with red speedup annotations and no thread references."""
    n_values = sorted(df['N'].unique())
    fig, axes = plt.subplots(1, len(n_values), figsize=(6*len(n_values), 7))
    if len(n_values) == 1:
        axes = [axes]
    
    for ax_idx, n in enumerate(n_values):
        subset = df[df['N'] == n].copy()
        subset['np'] = subset['PROW'] * subset['PCOL']
        
        ax = axes[ax_idx]
        
        # Sort by NP for a clean line
        subset = subset.sort_values('np')
        np_vals = subset['np'].values
        speedups = subset['matmatdist_speedup'].values
        
        # Plot single scaling line
        ax.plot(np_vals, speedups, marker='o', linewidth=2.5, markersize=8, 
                color='#FFA62B', label='matmatdist')
        
        # Annotate speedup values in red
        for x, y, val in zip(np_vals, speedups, speedups):
            ax.text(x, y + 0.05, f'{val:.2f}', color='red', fontsize=10, 
                    fontweight='bold', ha='center', va='bottom')
        
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Number of Processes (NP)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Speedup', fontsize=11, fontweight='bold')
        ax.set_title(f'N = {int(n)}\nDistributed Speedup Scaling', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_matmatdist_scaling_comparison(df, output_file='plots/matmatdist_scaling_comparison.png'):
    """Plot matmatdist scaling comparison for N=2048, 4096, and 6144."""
    target_n = [2048, 4096, 6144]
    target_threads = [1, 2, 4, 8]
    colors = ['#FFA62B', '#4ECDC4', '#FF6B6B', '#45B7D1']
    markers = ['o', 's', 'D', 'p']
    
    fig, axes = plt.subplots(1, 3, figsize=(27, 8))
    
    for ax_idx, n in enumerate(target_n):
        subset_n = df[df['N'] == n].copy()
        subset_n['threads'] = subset_n['NTROW'] * subset_n['NTCOL']
        subset_n['np'] = subset_n['PROW'] * subset_n['PCOL']
        
        ax = axes[ax_idx]
        all_np = set()
        
        for idx, t in enumerate(target_threads):
            sub = subset_n[subset_n['threads'] == t].sort_values('np')
            if sub.empty:
                continue
                
            np_vals = sub['np'].values
            speedups = sub['matmatdist_speedup'].values
            all_np.update(np_vals.tolist())
            
            # Plot line
            ax.plot(np_vals, speedups, marker=markers[idx], linewidth=2.5, markersize=8, 
                    color=colors[idx], label=f'NT={t}')
            
            # Annotate speedup values in black with alternating offsets to avoid overlap
            for i, (x, y, val) in enumerate(zip(np_vals, speedups, speedups)):
                # Alternate between putting text above and below the point
                offset = 0.08 if i % 2 == 0 else -0.15
                va = 'bottom' if i % 2 == 0 else 'top'
                
                ax.text(x, y + offset, f'{val:.2f}', color='black', fontsize=10, 
                        fontweight='bold', ha='center', va=va)
        
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Number of Processes (NP)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Speedup', fontsize=12, fontweight='bold')
        ax.set_title(f'matmatdist Scaling (N={n})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        if len(all_np) > 0:
            ax.set_xticks(sorted(list(all_np)))
        ax.legend(title='Threads per Process', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def print_summary(df):
    """Print results summary."""
    print("\n" + "="*120)
    print("RESULTS SUMMARY")
    print("="*120)
    
    for idx, row in df.iterrows():
        print(f"\nTest: N={int(row['N'])}, NTROW={int(row['NTROW'])}, NTCOL={int(row['NTCOL'])}, "
              f"DB={int(row['DB'])}, PROW={int(row['PROW'])}, PCOL={int(row['PCOL'])}")
        print("-" * 120)
        print(f"{'Method':<20} {'Time (s)':<15} {'GFlops':<15} {'Speedup':<15} {'Efficiency':<15}")
        print("-" * 120)
        
        for method in ['matmat_ikj', 'matmatblock', 'matmatthread', 'matmatdist']:
            time_val = row[f'{method}_time'] if f'{method}_time' in row else 0
            gflops_val = row[f'{method}_gflops'] if f'{method}_gflops' in row else 0
            speedup_val = row[f'{method}_speedup'] if f'{method}_speedup' in row else 0
            efficiency_val = row[f'{method}_efficiency'] if f'{method}_efficiency' in row else 0
            
            print(f"{method:<20} {time_val:<15.6f} {gflops_val:<15.3f} {speedup_val:<15.3f} {efficiency_val:<15.6f}")
    
    print("=" * 120 + "\n")

def main():
    """Main entry point."""
    import sys
    
    directory = sys.argv[1] if len(sys.argv) > 1 else '.'
    
    print(f"Reading result files from: {directory}")
    df = read_all_results(directory)
    
    if df is None or df.empty:
        print("No data to plot")
        return
    
    print(f"Loaded {len(df)} result file(s)")
    
    print_summary(df)
    plot_speedup(df)
    plot_efficiency(df)
    plot_speedup_vs_processes(df)
    plot_speedup_vs_threads(df)
    plot_speedup_dist_thread_map(df)
    plot_matmatdist_scaling_comparison(df)
    
    print("Plot generation complete!")

if __name__ == '__main__':
    main()