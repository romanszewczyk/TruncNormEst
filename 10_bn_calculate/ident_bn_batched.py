"""
Optimal b(n) Calculation - 500 Iterations
Find b(n) such that: 1 = (1/k) * sum(sqrt(sum_sq_dev / (n-b(n))))
Makes standard deviation estimator unbiased for N(0,1)
NO NON-ASCII CHARACTERS

CONFIGURATION:
- k = 2e8 samples per n value
- n range: [2, 100]
- 500 iterations with different random seeds
- Distribution: Standard Normal N(0,1)
- Solve for b(n) that gives mean(s) = 1
- Output: optimal_bn_500times.mat

MEMORY STRATEGY: Process n-values in batches to fit in 64GB RAM
- Batch size: 33 n-values at a time (3 batches total)
- RAM per batch: ~49 GB (fits in 64 GB with margin)
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.75'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

import jax
import jax.numpy as jnp
from jax import random, jit
import scipy.io as sio
import numpy as np
import time
import traceback
import subprocess
import sys
from functools import partial
from datetime import datetime
from scipy.optimize import brentq, minimize_scalar
import gc

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


def print_header():
    print("=" * 80)
    print("OPTIMAL b(n) Calculation - Standard Normal (500 ITERATIONS)")
    print("=" * 80)
    print("CONFIGURATION:")
    print("  - Distribution: N(0,1) - NO TRUNCATION")
    print("  - k = 2e8 samples per n value")
    print("  - n range: [2, 100]")
    print("  - 500 iterations with different seeds")
    print("  - Solve: 1 = mean(sqrt(sum_sq_dev / (n-b(n))))")
    print("  - Find optimal b(n) that makes estimator unbiased")
    print("  - Memory strategy: Process n-values in batches")
    print("=" * 80)
    print(f"JAX version: {jax.__version__}")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Logging to: optimal_bn_log.txt")


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
        print(f"Warning: Could not query GPU: {e}")
        return None, 0, 0, 0


@partial(jit, static_argnums=(1, 2))
def generate_and_compute_stats_chunked(key, n_samples, chunk_size):
    """Generate STANDARD NORMAL samples and compute squared deviations"""
    samples = random.normal(key, shape=(n_samples, chunk_size))
    col_means = jnp.mean(samples, axis=0)
    squared_devs = jnp.sum((samples - col_means[None, :]) ** 2, axis=0)
    return squared_devs


def process_n_value_chunked(n_val, k, key, chunk_size=50000):
    """Process single n value - STANDARD NORMAL"""
    n_chunks = (k + chunk_size - 1) // chunk_size
    results = []
    
    for i in range(n_chunks):
        key, subkey = random.split(key)
        current_chunk_size = min(chunk_size, k - i * chunk_size)
        chunk_result = generate_and_compute_stats_chunked(
            subkey, n_val, current_chunk_size
        )
        results.append(np.array(chunk_result))
        
        if i % 500 == 0:
            gc.collect()
    
    return np.concatenate(results)


def process_n_batch(n_values_batch, k, key, chunk_size=50000, verbose=True):
    """Process a batch of n values - STANDARD NORMAL"""
    assert k > 0, "k must be positive"
    assert all(n > 0 for n in n_values_batch), "All n values must be positive"
    
    Md_sqrT = np.zeros((len(n_values_batch), k))
    
    for ni, n_val in enumerate(n_values_batch):
        if verbose:
            progress = (ni + 1) / len(n_values_batch) * 100
            
            if ni % 10 == 0:
                _, used_mb, _, free_mb = get_gpu_memory()
                if used_mb > 0:
                    print(f"    n={n_val:3d} ({ni+1:3d}/{len(n_values_batch):3d}, {progress:5.1f}%) | GPU: {used_mb:5d}/{used_mb+free_mb:5d} MB")
                else:
                    print(f"    n={n_val:3d} ({ni+1:3d}/{len(n_values_batch):3d}, {progress:5.1f}%)")
            elif ni == len(n_values_batch) - 1:
                print(f"    n={n_val:3d} ({ni+1:3d}/{len(n_values_batch):3d}, {progress:5.1f}%)")
        
        key, subkey = random.split(key)
        Md_sqrT[ni, :] = process_n_value_chunked(
            n_val, k, subkey, chunk_size
        )
    
    return Md_sqrT


def determine_safe_chunk_size():
    _, used_mb, total_mb, free_mb = get_gpu_memory()
    
    if total_mb == 0:
        print("\nWarning: Cannot determine GPU memory, using default chunk size")
        return 100000
    
    usable_mb = free_mb * 0.60
    bytes_per_sample_at_n100 = 2 * 100 * 8 * 1.2
    max_chunk = int((usable_mb * 1024 * 1024) / bytes_per_sample_at_n100)
    safe_chunk = (max_chunk // 50000) * 50000
    
    if safe_chunk > 500000:
        safe_chunk = 500000
    elif safe_chunk < 50000:
        safe_chunk = 50000
    
    print(f"\nDetermining safe chunk size:")
    print(f"  Free memory: {free_mb} MB")
    print(f"  Usable (60%): {usable_mb:.0f} MB")
    print(f"  Max safe chunk: {max_chunk:,}")
    print(f"  Selected: {safe_chunk:,}")
    
    return safe_chunk


def check_memory_safety(chunk_size, n_max, compilation_mb=500):
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
    
    print(f"\nGPU Memory Safety Check:")
    print(f"  Current used:     {used_mb:6.0f} MB")
    print(f"  Estimated need:   {estimated_need:6.0f} MB")
    print(f"  Projected total:  {projected_total:6.0f} MB")
    print(f"  Safety limit:     {safety_limit:6.0f} MB (90% of total)")
    print(f"  Margin:           {margin:6.0f} MB")
    
    if not safe:
        print(f"  Status: UNSAFE - would exceed safe limit!")
        return False, int(margin)
    elif margin < 2000:
        print(f"  Status: MARGINAL - only {margin:.0f} MB margin")
        return True, int(margin)
    else:
        print(f"  Status: SAFE - {margin:.0f} MB margin")
        return True, int(margin)


def determine_n_batch_size(k, available_ram_gb=64):
    """Determine how many n-values can be processed at once"""
    bytes_per_float64 = 8
    safety_factor = 0.75  # Use 75% of available RAM for safety
    
    # Md_sqrT array size: (n_batch, k) * 8 bytes
    usable_bytes = available_ram_gb * 1024**3 * safety_factor
    n_batch_size = int(usable_bytes / (k * bytes_per_float64))
    
    # Ensure reasonable batch size
    if n_batch_size < 10:
        n_batch_size = 10
        print(f"\nWARNING: Very small batch size ({n_batch_size}). Consider reducing k.")
    
    return n_batch_size


def find_optimal_bn(n_val, sum_sq_devs):
    """
    Find optimal b(n) such that: mean(sqrt(sum_sq_devs / (n - b(n)))) = 1
    
    Parameters:
    - n_val: sample size
    - sum_sq_devs: array of sum of squared deviations
    
    Returns:
    - optimal b(n) value
    """
    
    def objective(bn):
        if bn >= n_val - 1e-10:
            return 1e10
        s_values = np.sqrt(sum_sq_devs / (n_val - bn))
        return (np.mean(s_values) - 1.0) ** 2
    
    result = minimize_scalar(
        objective,
        bounds=(0, n_val - 1e-6),
        method='bounded',
        options={'xatol': 1e-8}
    )
    
    return result.x


def run_single_iteration(iteration_num, k, n, chunk_size, n_batch_size, data_seed):
    """Run a single iteration with given seed, processing n-values in batches"""
    print(f"\n{'='*80}")
    print(f"ITERATION {iteration_num}/500 (seed={data_seed})")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    # Split n-values into batches
    n_batches = []
    for i in range(0, len(n), n_batch_size):
        n_batches.append(n[i:i+n_batch_size])
    
    print(f"\nProcessing {len(n)} n-values in {len(n_batches)} batches of ~{n_batch_size} values each")
    print(f"Expected RAM per batch: ~{(n_batch_size * k * 8)/(1024**3):.1f} GB")
    
    # Initialize results
    bn_values = np.zeros(len(n))
    verification = np.zeros(len(n))
    
    key = random.PRNGKey(data_seed)
    
    # Process each batch
    for batch_idx, n_batch in enumerate(n_batches):
        batch_start_idx = batch_idx * n_batch_size
        
        print(f"\n--- Batch {batch_idx+1}/{len(n_batches)}: n=[{n_batch[0]}, {n_batch[-1]}] ({len(n_batch)} values) ---")
        
        # Generate data for this batch
        batch_gen_start = time.time()
        key, subkey = random.split(key)
        Md_sqrT_batch = process_n_batch(n_batch, k, subkey, chunk_size=chunk_size, verbose=True)
        batch_gen_time = time.time() - batch_gen_start
        print(f"  Batch generation: {batch_gen_time:.1f} s")
        
        # Optimize b(n) for this batch
        print(f"  Optimizing b(n) for batch...")
        batch_opt_start = time.time()
        
        for i, n_val in enumerate(n_batch):
            global_idx = batch_start_idx + i
            sum_sq_devs = Md_sqrT_batch[i, :]
            bn_values[global_idx] = find_optimal_bn(n_val, sum_sq_devs)
            verification[global_idx] = np.mean(np.sqrt(sum_sq_devs / (n_val - bn_values[global_idx])))
            
            if i % 10 == 0 or i == len(n_batch) - 1:
                print(f"    n={n_val:3d}: b(n)={bn_values[global_idx]:8.5f}, mean(s)={verification[global_idx]:.8f}")
        
        batch_opt_time = time.time() - batch_opt_start
        print(f"  Batch optimization: {batch_opt_time:.1f} s")
        print(f"  Batch total time: {time.time() - batch_gen_start:.1f} s")
        
        # Free memory before next batch
        del Md_sqrT_batch
        gc.collect()
    
    iteration_time = time.time() - start_time
    print(f"\nTotal iteration time: {iteration_time:.1f} s")
    
    return {
        'bn_values': bn_values,
        'verification': verification,
        'time': iteration_time,
        'seed': data_seed
    }


def print_statistics(all_results):
    """Print statistical summary of all iterations"""
    print(f"\n{'='*80}")
    print("STATISTICAL SUMMARY")
    print(f"{'='*80}")
    
    # Extract b(n) arrays from all iterations
    all_bn = np.array([r['bn_values'] for r in all_results])  # Shape: (500, 99)
    
    # Calculate statistics for each n
    bn_mean = np.mean(all_bn, axis=0)
    bn_std = np.std(all_bn, axis=0)
    bn_min = np.min(all_bn, axis=0)
    bn_max = np.max(all_bn, axis=0)
    
    n_values = np.arange(2, 101)
    
    print("\nOptimal b(n) statistics (selected values):")
    print("\n  n  |  Mean b(n) | Std b(n) |  Min b(n) |  Max b(n) | n-mean(b(n))")
    print("-----+------------+----------+-----------+-----------+-------------")
    
    for i in [0, 3, 8, 18, 28, 48, 73, 98]:  # n=2,5,10,20,30,50,75,100
        if i < len(n_values):
            n_minus_bn = n_values[i] - bn_mean[i]
            print(f" {n_values[i]:3d} | {bn_mean[i]:10.6f} | {bn_std[i]:8.6f} | "
                  f"{bn_min[i]:9.6f} | {bn_max[i]:9.6f} | {n_minus_bn:11.6f}")
    
    print(f"\nTime statistics:")
    times = np.array([r['time'] for r in all_results])
    print(f"  Mean time per iteration: {np.mean(times):.1f} s")
    print(f"  Std time per iteration:  {np.std(times):.1f} s")
    print(f"  Min time: {np.min(times):.1f} s, Max time: {np.max(times):.1f} s")


def save_batch_results(all_results, k, n):
    """Save all batch results"""
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")
    
    n_iterations = len(all_results)
    
    # Extract all b(n) arrays
    all_bn = np.array([r['bn_values'] for r in all_results])  # Shape: (500, 99)
    times_all = np.array([r['time'] for r in all_results])
    seeds_all = np.array([r['seed'] for r in all_results])
    
    # Extract verification from all iterations
    all_verification = np.array([r['verification'] for r in all_results])  # Shape: (500, 99)
    
    # Statistics
    bn_mean = np.mean(all_bn, axis=0)
    bn_std = np.std(all_bn, axis=0)
    bn_min = np.min(all_bn, axis=0)
    bn_max = np.max(all_bn, axis=0)
    
    # Calculate n - b(n)
    n_minus_bn = n - bn_mean
    
    # Use last iteration's verification (all iterations verified during computation)
    verification = all_verification[-1, :]
    
    # Mean verification across all iterations
    verification_mean = np.mean(all_verification, axis=0)
    verification_std = np.std(all_verification, axis=0)
    
    # Save to MAT file
    try:
        save_dict = {
            'k': k,
            'n': n,
            'n_iterations': n_iterations,
            # All iterations: (500, 99) matrix
            'bn_all': all_bn,
            'times_all': times_all,
            'seeds_all': seeds_all,
            # Statistics for each n: (99,) arrays
            'bn_mean': bn_mean,
            'bn_std': bn_std,
            'bn_min': bn_min,
            'bn_max': bn_max,
            # Derived quantities
            'n_minus_bn': n_minus_bn,
            'verification': verification,
            'verification_mean': verification_mean,
            'verification_std': verification_std,
            'all_verification': all_verification,
            # Additional info
            'description': 'Optimal b(n) such that 1 = mean(sqrt(sum_sq_dev/(n-b(n)))), 500 iterations, k=2e8 samples. Verification computed per iteration.'
        }
        sio.savemat('optimal_bn_500times.mat', save_dict, do_compression=True)
        print("  Saved: optimal_bn_500times.mat")
        print("\n  MAT file contents:")
        print("    n: array of n values (2 to 100)")
        print("    bn_all: (500 x 99) matrix of all optimal b(n) values")
        print("    bn_mean: mean optimal b(n) for each n")
        print("    bn_std: standard deviation of b(n) for each n")
        print("    n_minus_bn: array of n - mean(b(n)) values")
        print("    verification: mean(s) from last iteration (should all be ~1.0)")
        print("    verification_mean: mean verification across all 500 iterations")
        print("    all_verification: (500 x 99) matrix of all verification values")
        
        print(f"\n  Verification statistics (should all be ~1.000):")
        for i in [0, 8, 18, 48, 98]:
            if i < len(n):
                print(f"    n={n[i]:3d}: mean={verification_mean[i]:.8f}, std={verification_std[i]:.8f}")
                
    except Exception as e:
        print(f"  ERROR saving MAT: {e}")


def main():
    # Initialize logger
    logger = Logger('optimal_bn_log.txt')
    sys.stdout = logger
    
    try:
        print_header()
        
        gpu_name, gpu_used_mb, gpu_total_mb, gpu_free_mb = get_gpu_memory()
        
        if gpu_name is None:
            print("\nWarning: GPU not detected or nvidia-smi not available")
            print("Running on CPU (will be slower)")
            gpu_total_mb = 24576
            gpu_free_mb = 20000
            gpu_used_mb = gpu_total_mb - gpu_free_mb
        else:
            print(f"\nGPU: {gpu_name}")
            print(f"Memory at startup:")
            print(f"  Used:  {gpu_used_mb:5d} MB ({100*gpu_used_mb/gpu_total_mb:5.1f}%)")
            print(f"  Free:  {gpu_free_mb:5d} MB ({100*gpu_free_mb/gpu_total_mb:5.1f}%)")
            print(f"  Total: {gpu_total_mb:5d} MB")
        
        print("=" * 80)
        print()
        
        k = int(2e8)
        n = np.arange(2, 101)
        n_iterations = 500
        
        # Determine batch size for n-values based on available RAM
        n_batch_size = determine_n_batch_size(k, available_ram_gb=64)
        n_num_batches = int(np.ceil(len(n) / n_batch_size))
        
        print(f"\nBatch Parameters:")
        print(f"  k = {k:,} samples per n value")
        print(f"  n range: [2, 100] ({len(n)} points)")
        print(f"  Number of iterations: {n_iterations}")
        print(f"  Distribution: Standard Normal N(0,1) - NO TRUNCATION")
        print(f"  Goal: Find b(n) such that mean(sqrt(sum_sq_dev/(n-b(n)))) = 1")
        print(f"\n  MEMORY STRATEGY:")
        print(f"  - Process n-values in {n_num_batches} batches of {n_batch_size} values each")
        print(f"  - RAM per batch: ~{(n_batch_size * k * 8)/(1024**3):.1f} GB (fits in 64 GB)")
        print(f"  - Total n-values: {len(n)}")
        
        chunk_size = determine_safe_chunk_size()
        n_max = max(n)
        safe, margin = check_memory_safety(chunk_size, n_max)
        
        if not safe:
            print("\nERROR: Not enough free GPU memory!")
            return
        
        start_time_total = time.time()
        all_results = []
        
        # Run iterations
        for iteration in range(1, n_iterations + 1):
            data_seed = 42 + iteration
            result = run_single_iteration(iteration, k, n, chunk_size, n_batch_size, data_seed)
            all_results.append(result)
            
            if iteration % 10 == 0:
                elapsed = time.time() - start_time_total
                avg_time = elapsed / iteration
                remaining = avg_time * (n_iterations - iteration)
                print(f"\n{'='*80}")
                print(f"Progress: {iteration}/{n_iterations} ({100*iteration/n_iterations:.1f}%)")
                print(f"  Elapsed: {elapsed/60:.1f} min, Estimated remaining: {remaining/60:.1f} min")
                print(f"  Estimated total: {(elapsed + remaining)/60:.1f} min")
                print(f"{'='*80}")
        
        # Print statistics
        print_statistics(all_results)
        
        # Save results
        save_batch_results(all_results, k, n)
        
        total_time = time.time() - start_time_total
        print(f"\n{'='*80}")
        print(f"COMPLETED! Total: {total_time/60:.2f} minutes")
        print(f"Average per iteration: {total_time/n_iterations:.1f} seconds")
        print(f"{'='*80}")
        print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nOutput files:")
        print("  - optimal_bn_500times.mat")
        print("  - optimal_bn_log.txt")
        
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
