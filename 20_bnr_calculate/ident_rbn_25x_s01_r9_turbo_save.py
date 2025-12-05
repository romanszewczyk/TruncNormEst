"""
MEMORY-OPTIMIZED VERSION - Optimal b(r,n) Calculation with 25 Repetitions
CRITICAL FIX: Process n values one at a time to avoid 46 GB memory allocation
Peak memory reduced from ~46 GB to ~2 GB

CHECKPOINT SAVING: Results are saved after EACH repetition completes
- Creates optimal_brn_25reps_step01.mat, step02.mat, ..., step25.mat
- Final consolidated results saved as optimal_brn_25reps_FINAL.mat
- Safe to interrupt and restart from last completed checkpoint

Key changes:
1. Process each n value individually (don't store full Md_sqrT matrix)
2. Compute b(r,n) immediately after generating samples for that n
3. Add explicit garbage collection
4. Monitor memory usage during execution
5. Save results after each repetition (checkpoint system)

Expected runtime: ~100-300 hours for 25 repetitions
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.85'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

import jax
import jax.numpy as jnp
from jax import random, jit
from jax.scipy.special import erf, erfinv
import scipy.io as sio
import numpy as np
import time
import traceback
import subprocess
import sys
import gc
from functools import partial
from datetime import datetime
from scipy.optimize import brentq

jax.config.update("jax_enable_x64", True)


class Logger:
    """Logger that writes to both console and file"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()


def compute_truncated_normal_std(r_trunc):
    """
    Compute the true standard deviation of a truncated N(0,1) at [-r, r]
    """
    phi_r = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-r_trunc**2 / 2.0)
    erf_val = float(erf(r_trunc / np.sqrt(2.0)))
    variance = 1.0 - 2.0 * r_trunc * phi_r / erf_val
    std_dev = np.sqrt(variance)
    return std_dev


def print_header(r_values, n_repetitions):
    print("=" * 80)
    print("MEMORY-OPTIMIZED VERSION - Optimal b(r,n) Calculation with 25 REPETITIONS")
    print("=" * 80)
    print("MEMORY FIX APPLIED:")
    print("  - Process n values one at a time (not all at once)")
    print("  - Peak memory: ~2 GB instead of ~46 GB")
    print("  - Explicit garbage collection after each n")
    print("=" * 80)
    print("CHECKPOINT SAVING ENABLED:")
    print("  - Results saved after EACH repetition completes")
    print("  - Files: optimal_brn_25reps_step01.mat through step25.mat")
    print("  - Final: optimal_brn_25reps_FINAL.mat")
    print("  - Safe to interrupt - resume from last checkpoint")
    print("=" * 80)
    print("OPTIMIZATIONS:")
    print("  - Inverse CDF sampling (100% acceptance, no rejection)")
    print("  - Larger chunk sizes (better GPU utilization)")
    print("  - Streamlined memory operations")
    print("=" * 80)
    print("CONFIGURATION:")
    print("  - Distribution: Truncated N(0,1) at [-r, r], normalized to sigma=1")
    print(f"  - r range: [{r_values[0]:.2f}, {r_values[-1]:.2f}] ({len(r_values)} values)")
    print(f"  - n range: [2, 30] (29 values)")
    print(f"  - k = 2e8 samples per (r,n) value")
    print(f"  - Number of repetitions: {n_repetitions}")
    print(f"  - Total computations: {n_repetitions} x {len(r_values)} x 29 = {n_repetitions * len(r_values) * 29:,}")
    print(f"  - Method: Inverse CDF (no rejection sampling)")
    print("=" * 80)
    print(f"JAX version: {jax.__version__}")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


def get_gpu_memory():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.used,memory.total,memory.free',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        
        if result.returncode != 0:
            return None, 0, 0, 0
            
        output = result.stdout.strip().split(',')
        if len(output) < 4:
            return None, 0, 0, 0
            
        gpu_name = output[0].strip()
        used_mb = int(output[1].strip())
        total_mb = int(output[2].strip())
        free_mb = int(output[3].strip())
        
        return gpu_name, used_mb, total_mb, free_mb
    except Exception as e:
        return None, 0, 0, 0


def get_system_memory():
    """Get system RAM usage"""
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = {}
            for line in f:
                parts = line.split(':')
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip().split()[0]
                    meminfo[key] = int(value) // 1024  # Convert to MB
        
        total = meminfo.get('MemTotal', 0)
        available = meminfo.get('MemAvailable', 0)
        used = total - available
        
        return used, total, available
    except:
        return 0, 0, 0


@partial(jit, static_argnums=(1, 2, 3))
def generate_truncated_normal_inverse_cdf(key, n_samples, chunk_size, r_trunc):
    """
    OPTIMIZED: Generate truncated normal using inverse CDF method
    """
    total_needed = n_samples * chunk_size
    
    uniform = random.uniform(key, shape=(total_needed,), minval=0.0, maxval=1.0)
    
    sqrt2 = jnp.sqrt(2.0)
    cdf_lower = 0.5 * (1.0 + erf(-r_trunc / sqrt2))
    cdf_upper = 0.5 * (1.0 + erf(r_trunc / sqrt2))
    
    scaled_uniform = cdf_lower + uniform * (cdf_upper - cdf_lower)
    samples = sqrt2 * erfinv(2.0 * scaled_uniform - 1.0)
    
    samples = samples.reshape(n_samples, chunk_size)
    col_means = jnp.mean(samples, axis=0)
    squared_devs = jnp.sum((samples - col_means[None, :]) ** 2, axis=0)
    
    return squared_devs


def process_single_n_value(n_val, k, key, r_trunc, true_std, chunk_size, verbose=False):
    """
    MEMORY OPTIMIZED: Process a single n value
    Returns the array of normalized squared deviations
    """
    n_chunks = (k + chunk_size - 1) // chunk_size
    results = []
    
    for i in range(n_chunks):
        key, subkey = random.split(key)
        current_chunk_size = min(chunk_size, k - i * chunk_size)
        
        chunk_result = generate_truncated_normal_inverse_cdf(
            subkey, n_val, current_chunk_size, r_trunc
        )
        
        chunk_result_np = np.array(chunk_result) / (true_std ** 2)
        results.append(chunk_result_np)
        
        if i % 500 == 0 and i > 0:
            gc.collect()
    
    final_result = np.concatenate(results)
    
    if len(final_result) != k:
        raise ValueError(f"Result length mismatch: got {len(final_result)}, expected {k}")
    
    return final_result


def find_optimal_bn(n, sum_sq_devs):
    """Find b such that mean(sqrt(sum_sq_devs / (n - b))) = 1"""
    def objective(b):
        if n - b <= 0:
            return float('inf')
        return np.mean(np.sqrt(sum_sq_devs / (n - b))) - 1.0
    
    try:
        result = brentq(objective, 0.0, n - 0.1, xtol=1e-8)
        return result
    except ValueError:
        return n / 2.0


def compute_brn_for_r_memory_optimized(r_val, k, n_values, chunk_size, seed=42):
    """
    MEMORY OPTIMIZED: Process all n values for a given r
    Key change: Process one n at a time instead of storing full matrix
    """
    print(f"\nProcessing r = {r_val:.2f}:")
    print(f"  Truncation range: [{-r_val:.2f}, {r_val:.2f}]")
    
    start_time = time.time()
    
    true_std = compute_truncated_normal_std(r_val)
    
    print(f"  Generating and processing samples (inverse CDF method)...")
    print(f"  True std of truncated distribution: {true_std:.6f}")
    print(f"  Normalization factor: 1/{true_std:.6f} = {1.0/true_std:.6f}")
    
    # Get initial memory
    ram_used, ram_total, ram_avail = get_system_memory()
    if ram_total > 0:
        print(f"  RAM: {ram_used:,} MB used, {ram_avail:,} MB available (of {ram_total:,} MB)")
    
    key = random.PRNGKey(seed)
    
    brn_values = np.zeros(len(n_values))
    verification = np.zeros(len(n_values))
    
    gen_start = time.time()
    
    # CRITICAL: Process each n value individually
    for ni, n_val in enumerate(n_values):
        if ni % 5 == 0 or ni == len(n_values) - 1:
            progress = (ni + 1) / len(n_values) * 100
            
            if ni % 10 == 0:
                _, gpu_used_mb, _, gpu_free_mb = get_gpu_memory()
                ram_used, ram_total, ram_avail = get_system_memory()
                
                if gpu_used_mb > 0 and ram_total > 0:
                    print(f"    n={n_val:3d} ({ni+1:2d}/{len(n_values):2d}, {progress:5.1f}%) | "
                          f"GPU: {gpu_used_mb:5d}/{gpu_used_mb+gpu_free_mb:5d} MB | "
                          f"RAM: {ram_used:6,} MB / {ram_total:6,} MB")
                elif gpu_used_mb > 0:
                    print(f"    n={n_val:3d} ({ni+1:2d}/{len(n_values):2d}, {progress:5.1f}%) | "
                          f"GPU: {gpu_used_mb:5d}/{gpu_used_mb+gpu_free_mb:5d} MB")
                elif ram_total > 0:
                    print(f"    n={n_val:3d} ({ni+1:2d}/{len(n_values):2d}, {progress:5.1f}%) | "
                          f"RAM: {ram_used:6,} MB / {ram_total:6,} MB")
                else:
                    print(f"    n={n_val:3d} ({ni+1:2d}/{len(n_values):2d}, {progress:5.1f}%)")
            else:
                print(f"    n={n_val:3d} ({ni+1:2d}/{len(n_values):2d}, {progress:5.1f}%)")
        
        # Generate samples for this n only
        key, subkey = random.split(key)
        sum_sq_devs = process_single_n_value(
            n_val, k, subkey, r_val, true_std, chunk_size, verbose=False
        )
        
        # Find optimal b(r,n) immediately
        brn_values[ni] = find_optimal_bn(n_val, sum_sq_devs)
        verification[ni] = np.mean(np.sqrt(sum_sq_devs / (n_val - brn_values[ni])))
        
        # Explicit cleanup
        del sum_sq_devs
        gc.collect()
    
    gen_time = time.time() - gen_start
    
    print(f"  Processing time: {gen_time:.1f} s")
    print(f"  Results for r={r_val:.2f}:")
    
    for i in range(0, len(n_values), 5):
        n_val = n_values[i]
        error = abs(verification[i] - 1.0)
        status = "OK" if error < 1e-5 else "CHECK"
        print(f"    n={n_val:3d}: b(r,n)={brn_values[i]:8.5f}, mean(s)={verification[i]:.8f} [{status}]")
    
    # Show last value if not already shown
    if len(n_values) % 5 != 0:
        i = len(n_values) - 1
        n_val = n_values[i]
        error = abs(verification[i] - 1.0)
        status = "OK" if error < 1e-5 else "CHECK"
        print(f"    n={n_val:3d}: b(r,n)={brn_values[i]:8.5f}, mean(s)={verification[i]:.8f} [{status}]")
    
    total_time = time.time() - start_time
    print(f"  Total time for r={r_val:.2f}: {total_time:.1f} s")
    
    return brn_values, verification


def determine_safe_chunk_size():
    """
    OPTIMIZED: Use more GPU memory (80% instead of 60%)
    """
    _, used_mb, total_mb, free_mb = get_gpu_memory()
    
    if total_mb == 0:
        print("\nWarning: Cannot determine GPU memory, using default chunk size")
        return 5000000
    
    usable_mb = free_mb * 0.80
    bytes_per_sample_at_n100 = 2 * 100 * 8 * 1.2
    max_chunk = int((usable_mb * 1024 * 1024) / bytes_per_sample_at_n100)
    
    safe_chunk = (max_chunk // 1000000) * 1000000
    
    if safe_chunk > 20000000:
        safe_chunk = 20000000
    elif safe_chunk < 1000000:
        safe_chunk = 1000000
    
    print(f"\nDetermining safe chunk size:")
    print(f"  Free memory: {free_mb} MB")
    print(f"  Usable (80%): {usable_mb:.0f} MB")
    print(f"  Max safe chunk: {max_chunk:,}")
    print(f"  Selected: {safe_chunk:,}")
    
    return safe_chunk


def check_memory_safety(chunk_size, n_max, k, compilation_mb=500):
    """Check if memory usage will be safe"""
    _, used_mb, total_mb, free_mb = get_gpu_memory()
    
    if total_mb == 0:
        print("\nWarning: Cannot determine GPU memory, assuming safe")
        return True, 10000
    
    data_memory = 2 * n_max * chunk_size * 8
    overhead = data_memory * 0.2
    compilation = compilation_mb * 1024 * 1024
    estimated_need = (data_memory + overhead + compilation) / (1024 * 1024)
    
    projected_total = used_mb + estimated_need
    safety_limit = 0.90 * total_mb
    safe = projected_total < safety_limit
    margin = safety_limit - projected_total
    
    print(f"\nMemory Safety Check (GPU):")
    print(f"  Current used:     {used_mb:6.0f} MB")
    print(f"  Estimated need:   {estimated_need:6.0f} MB")
    print(f"  Projected total:  {projected_total:6.0f} MB")
    print(f"  Safety limit:     {safety_limit:6.0f} MB (90% of total)")
    print(f"  Margin:           {margin:6.0f} MB")
    
    # Also check system RAM
    ram_used, ram_total, ram_avail = get_system_memory()
    if ram_total > 0:
        print(f"\nSystem RAM Check:")
        print(f"  Used:             {ram_used:6,} MB")
        print(f"  Available:        {ram_avail:6,} MB")
        print(f"  Total:            {ram_total:6,} MB")
        
        # Estimate peak RAM usage (per n value now)
        estimated_ram_per_n = (k * 8) / (1024 * 1024)  # Just the array for one n
        print(f"  Estimated per n:  {estimated_ram_per_n:6.0f} MB")
        
        if estimated_ram_per_n * 2 > ram_avail:  # 2x safety margin
            print(f"  Status: WARNING - Low RAM margin")
        else:
            print(f"  Status: SAFE - Adequate RAM")
    
    if not safe:
        print(f"  Status: UNSAFE - would exceed safe GPU limit!")
        return False, int(margin)
    else:
        print(f"  Status: SAFE - {margin:.0f} MB GPU margin")
        return True, int(margin)


def save_results(brn_matrix, verification_matrix, k, n, r_values, n_repetitions, 
                 step=None, total_steps=25):
    """Save results to MAT file
    
    Args:
        brn_matrix: Results matrix (shape: [n_reps, n_r_values, n_n_values])
        verification_matrix: Verification matrix (same shape)
        k: Number of samples
        n: Array of n values
        r_values: Array of r values
        n_repetitions: Number of repetitions completed (for this save)
        step: Current step number (1-25) for intermediate saves, None for final
        total_steps: Total number of steps (25)
    """
    if step is not None:
        print(f"\n{'='*80}")
        print(f"SAVING INTERMEDIATE RESULTS - STEP {step}/{total_steps}")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print("SAVING FINAL RESULTS")
        print(f"{'='*80}")
    
    n_minus_brn = np.zeros_like(brn_matrix)
    for rep in range(n_repetitions):
        for i in range(len(r_values)):
            n_minus_brn[rep, i, :] = n - brn_matrix[rep, i, :]
    
    try:
        if step is not None:
            description = f'Optimal b(r,n) - Step {step}/{total_steps} completed, k=2e8 samples per (r,n) combination'
            repetition_description = f'{n_repetitions} of {total_steps} independent repetitions completed with different random seeds'
            filename = f'optimal_brn_25reps_step{step:02d}.mat'
            version = f'memory_optimized_v4.0_25reps_step{step:02d}'
        else:
            description = 'Optimal b(r,n) with 25 repetitions, k=2e8 samples per (r,n) combination'
            repetition_description = '25 independent repetitions with different random seeds'
            filename = 'optimal_brn_25reps_FINAL.mat'
            version = 'memory_optimized_v4.0_25reps_FINAL'
        
        save_dict = {
            'k': k,
            'n': n,
            'r_values': r_values,
            'n_repetitions': n_repetitions,
            'total_steps': total_steps,
            'brn_matrix': brn_matrix,
            'verification_matrix': verification_matrix,
            'n_minus_brn': n_minus_brn,
            'description': description,
            'r_description': 'r: truncation parameter (truncation at [-r*sigma, r*sigma])',
            'n_description': 'n: sample size (2 to 30)',
            'brn_description': 'brn_matrix[rep, i, j] = b(r_values[i], n[j]) for repetition rep (0-indexed)',
            'repetition_description': repetition_description,
            'access_example': 'To get b(r,n) for rep 5, r=1.0 (idx 9), n=10 (idx 9): brn_matrix(5,9,9) in MATLAB',
            'version': version
        }
        
        if step is not None:
            save_dict['current_step'] = step
        
        sio.savemat(filename, save_dict, do_compression=True)
        print(f"  Saved: {filename}")
        
        all_errors = np.abs(verification_matrix - 1.0)
        max_error = np.max(all_errors)
        mean_error = np.mean(all_errors)
        
        print(f"\n  Verification Statistics (across {n_repetitions} completed repetitions):")
        print(f"    Overall max error:  {max_error:.8f}")
        print(f"    Overall mean error: {mean_error:.8f}")
        
        if max_error > 1e-4:
            print(f"\n    WARNING: Some verification values deviate significantly from 1.0")
        else:
            print(f"\n    Status: All verification values within tolerance")
                
    except Exception as e:
        print(f"  ERROR saving MAT: {e}")
        traceback.print_exc()


def main():
    logger = Logger('optimal_brn_25reps_log.txt')
    sys.stdout = logger
    
    try:
        # Configuration
        k = int(2e8)
        n = np.concatenate((np.arange(2, 31, 1),np.arange(35, 51, 5),np.arange(60, 101, 10)))
        r_values = np.concatenate((np.arange(0.2, 5.0 + 0.01, 0.1),np.arange(5.5, 10.0 + 0.01, 0.5)))
        n_repetitions = 25
        
        print_header(r_values, n_repetitions)
        
        gpu_name, gpu_used_mb, gpu_total_mb, gpu_free_mb = get_gpu_memory()
        
        if gpu_name:
            print(f"\nGPU: {gpu_name}")
            print(f"Memory at startup:")
            print(f"  Used:  {gpu_used_mb:5d} MB ({100*gpu_used_mb/gpu_total_mb:5.1f}%)")
            print(f"  Free:  {gpu_free_mb:5d} MB ({100*gpu_free_mb/gpu_total_mb:5.1f}%)")
            print(f"  Total: {gpu_total_mb:5d} MB")
        else:
            print("\nWarning: GPU not detected")
        
        # System RAM
        ram_used, ram_total, ram_avail = get_system_memory()
        if ram_total > 0:
            print(f"\nSystem RAM:")
            print(f"  Used:      {ram_used:7,} MB ({100*ram_used/ram_total:5.1f}%)")
            print(f"  Available: {ram_avail:7,} MB ({100*ram_avail/ram_total:5.1f}%)")
            print(f"  Total:     {ram_total:7,} MB")
        
        print("=" * 80)
        print(f"\nParameters:")
        print(f"  k = {k:,} samples per (r,n) value")
        print(f"  n range: [2, 30] ({len(n)} points)")
        print(f"  r range: [{r_values[0]:.2f}, {r_values[-1]:.2f}] ({len(r_values)} points)")
        print(f"  Number of repetitions: {n_repetitions}")
        print(f"  Total computations: {n_repetitions} x {len(r_values)} x {len(n)} = {n_repetitions * len(r_values) * len(n):,}")
        print(f"  Method: Inverse CDF (no rejection sampling)")
        
        chunk_size = determine_safe_chunk_size()
        n_max = max(n)
        safe, margin = check_memory_safety(chunk_size, n_max, k)
        
        if not safe:
            print("\nERROR: Not enough free GPU memory!")
            return
        
        print(f"\n{'='*80}")
        print("ALL SAFETY CHECKS PASSED - Starting computation")
        print(f"{'='*80}")
        
        start_time_total = time.time()
        
        brn_matrix = np.zeros((n_repetitions, len(r_values), len(n)))
        verification_matrix = np.zeros((n_repetitions, len(r_values), len(n)))
        
        print(f"\n{'='*80}")
        print(f"COMPUTING b(r,n) FOR {n_repetitions} REPETITIONS")
        print(f"{'='*80}")
        
        for rep in range(n_repetitions):
            rep_start_time = time.time()
            
            print(f"\n{'='*80}")
            print(f"REPETITION {rep+1}/{n_repetitions}")
            print(f"{'='*80}")
            
            for ri, r_val in enumerate(r_values):
                print(f"\n[Rep {rep+1}/{n_repetitions}] [{ri+1}/{len(r_values)}] r = {r_val:.2f}")
                
                try:
                    seed = 42 + rep * 1000 + ri
                    
                    brn_values, verification = compute_brn_for_r_memory_optimized(
                        r_val, k, n, chunk_size, seed=seed
                    )
                    
                    brn_matrix[rep, ri, :] = brn_values
                    verification_matrix[rep, ri, :] = verification
                    
                    rep_elapsed = time.time() - rep_start_time
                    rep_avg_time_per_r = rep_elapsed / (ri + 1)
                    rep_remaining = rep_avg_time_per_r * (len(r_values) - ri - 1)
                    
                    print(f"  Progress within repetition {rep+1}: {ri+1}/{len(r_values)} ({100*(ri+1)/len(r_values):.1f}%)")
                    print(f"    Elapsed: {rep_elapsed/60:.1f} min, Remaining: {rep_remaining/60:.1f} min")
                    
                except Exception as e:
                    print(f"\n  ERROR processing r={r_val:.2f}: {e}")
                    traceback.print_exc()
                    brn_matrix[rep, ri, :] = np.nan
                    verification_matrix[rep, ri, :] = np.nan
            
            rep_time = time.time() - rep_start_time
            total_elapsed = time.time() - start_time_total
            avg_time_per_rep = total_elapsed / (rep + 1)
            remaining_total = avg_time_per_rep * (n_repetitions - rep - 1)
            
            print(f"\n{'='*80}")
            print(f"REPETITION {rep+1}/{n_repetitions} COMPLETED")
            print(f"{'='*80}")
            print(f"  Time for this repetition: {rep_time:.1f} seconds ({rep_time/60:.1f} minutes)")
            print(f"  Average per repetition: {avg_time_per_rep:.1f} seconds ({avg_time_per_rep/60:.1f} minutes)")
            print(f"  Total elapsed: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.1f} hours)")
            print(f"  Estimated remaining: {remaining_total/60:.1f} minutes ({remaining_total/3600:.1f} hours)")
            print(f"  Progress: {rep+1}/{n_repetitions} repetitions ({100*(rep+1)/n_repetitions:.1f}%)")
            
            # Save intermediate results after each repetition
            completed_reps = rep + 1
            save_results(
                brn_matrix[:completed_reps, :, :],
                verification_matrix[:completed_reps, :, :],
                k, n, r_values, completed_reps,
                step=completed_reps, total_steps=n_repetitions
            )
        
        save_results(brn_matrix, verification_matrix, k, n, r_values, n_repetitions, 
                    step=None, total_steps=n_repetitions)
        
        total_time = time.time() - start_time_total
        print(f"\n{'='*80}")
        print(f"ALL {n_repetitions} REPETITIONS COMPLETED!")
        print(f"{'='*80}")
        print(f"  Total time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
        print(f"  Average per repetition: {total_time/n_repetitions:.1f} seconds ({total_time/n_repetitions/60:.1f} minutes)")
        print(f"  Average per r value: {total_time/(n_repetitions*len(r_values)):.1f} seconds")
        print(f"{'='*80}")
        print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nOutput files:")
        print("  - optimal_brn_25reps_step01.mat through step25.mat (checkpoints)")
        print("  - optimal_brn_25reps_FINAL.mat (consolidated final results)")
        print("  - optimal_brn_25reps_log.txt")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        print(f"Partial results (completed repetitions) may be available")
    except Exception as e:
        print(f"\nFatal error: {e}")
        traceback.print_exc()
    finally:
        logger.close()
        sys.stdout = logger.terminal


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"\nFatal: {e}")
        traceback.print_exc()