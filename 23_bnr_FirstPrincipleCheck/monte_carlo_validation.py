"""
Monte Carlo validation of truncated normal standard deviation estimators.

This script validates the optimal b(n,r) estimator against the baseline
alpha(r)-only estimator through large-scale Monte Carlo simulation.

Features:
- GPU acceleration via JAX
- Chunked processing for arbitrary k values (1e6 to 1e9+)
- Memory-efficient computation
- Progress tracking and timing
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
from jax import jit, vmap
import time
from typing import Dict, List, Tuple
import gc

from truncated_normal_estimator import TruncatedNormalEstimator
from truncated_normal_generator import TruncatedNormalGenerator, truncnorm_simple


class MonteCarloValidator:
    """
    Monte Carlo validation framework with chunking strategy.
    """
    
    def __init__(self, use_gpu: bool = True, chunk_size: int = 100000):
        """
        Initialize the validator.
        
        Parameters
        ----------
        use_gpu : bool
            Whether to use GPU acceleration
        chunk_size : int
            Number of Monte Carlo trials per chunk
            Adjust based on available memory
        """
        self.use_gpu = use_gpu
        self.chunk_size = chunk_size
        
        # Initialize components
        self.estimator = TruncatedNormalEstimator(use_gpu=use_gpu)
        self.generator = TruncatedNormalGenerator(use_gpu=use_gpu)
        
        # Compile the core Monte Carlo function
        self._compile_monte_carlo()
        
        print(f"MonteCarloValidator initialized")
        print(f"  GPU enabled: {self.use_gpu}")
        print(f"  Chunk size: {self.chunk_size:,}")
        if self.use_gpu:
            print(f"  GPU devices: {jax.devices('gpu')}")
    
    def _compile_monte_carlo(self):
        """Compile the core Monte Carlo computation."""
        
        @jit
        def process_chunk_opti(samples: jnp.ndarray, r: float,
                               nn: jnp.ndarray, rr: jnp.ndarray,
                               bnr: jnp.ndarray) -> jnp.ndarray:
            """Process a chunk with optimal estimator."""
            n, k = samples.shape
            
            # Compute means
            mu = jnp.mean(samples, axis=0)
            
            # Center data
            xc = samples - mu
            
            # Interpolate b(n,r)
            # Find indices for interpolation
            i = jnp.searchsorted(rr, r) - 1
            j = jnp.searchsorted(nn, float(n)) - 1
            
            i = jnp.clip(i, 0, len(rr) - 2)
            j = jnp.clip(j, 0, len(nn) - 2)
            
            x1, x2 = nn[j], nn[j + 1]
            y1, y2 = rr[i], rr[i + 1]
            
            q11 = bnr[i, j]
            q12 = bnr[i + 1, j]
            q21 = bnr[i, j + 1]
            q22 = bnr[i + 1, j + 1]
            
            fxy1 = ((x2 - n) * q11 + (n - x1) * q21) / (x2 - x1)
            fxy2 = ((x2 - n) * q12 + (n - x1) * q22) / (x2 - x1)
            bn = ((y2 - r) * fxy1 + (r - y1) * fxy2) / (y2 - y1)
            
            # Compute alpha(r)
            phi_r = jnp.exp(-0.5 * r**2) / jnp.sqrt(2 * jnp.pi)
            Phi_r = 0.5 * (1.0 + jax.scipy.special.erf(r / jnp.sqrt(2.0)))
            ar = 1.0 - 2.0 * r * phi_r / (2.0 * Phi_r - 1.0)
            
            # Compute estimates
            ss = jnp.sum(xc**2, axis=0)
            s_est = jnp.sqrt(ss / ((n - bn) * ar))
            
            return s_est
        
        @jit
        def process_chunk_baseline(samples: jnp.ndarray, r: float) -> jnp.ndarray:
            """Process a chunk with baseline estimator."""
            n, k = samples.shape
            
            # Compute means
            mu = jnp.mean(samples, axis=0)
            
            # Center data
            xc = samples - mu
            
            # Use bn = 1 (no b(n,r) correction)
            bn = 1.0
            
            # Compute alpha(r)
            phi_r = jnp.exp(-0.5 * r**2) / jnp.sqrt(2 * jnp.pi)
            Phi_r = 0.5 * (1.0 + jax.scipy.special.erf(r / jnp.sqrt(2.0)))
            ar = 1.0 - 2.0 * r * phi_r / (2.0 * Phi_r - 1.0)
            
            # Compute estimates
            ss = jnp.sum(xc**2, axis=0)
            s_est = jnp.sqrt(ss / ((n - bn) * ar))
            
            return s_est
        
        @jit
        def generate_and_estimate_chunk(key: random.PRNGKey, 
                                       n: int, r: float, 
                                       chunk_size: int,
                                       nn: jnp.ndarray, 
                                       rr: jnp.ndarray,
                                       bnr: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Generate samples and compute both estimates in one pass."""
            # Generate truncated normal samples
            # For efficiency, use rejection sampling
            key1, key2 = random.split(key)
            
            # Allocate sample array
            samples = jnp.zeros((n, chunk_size))
            
            # Generate samples (simplified for vectorization)
            max_attempts = 10
            for i in range(n):
                key_i, key1 = random.split(key1)
                
                # Generate with rejection
                accepted = 0
                for attempt in range(max_attempts):
                    key_attempt, key_i = random.split(key_i)
                    candidates = random.normal(key_attempt, (chunk_size * 2,))
                    mask = (candidates >= -r) & (candidates <= r)
                    valid = candidates[mask]
                    
                    n_valid = jnp.minimum(len(valid), chunk_size - accepted)
                    samples = samples.at[i, accepted:accepted+n_valid].set(valid[:n_valid])
                    accepted += n_valid
                    
                    if accepted >= chunk_size:
                        break
            
            # Compute both estimates
            s_opti = process_chunk_opti(samples, r, nn, rr, bnr)
            s_baseline = process_chunk_baseline(samples, r)
            
            return s_opti, s_baseline
        
        self.process_chunk_opti = process_chunk_opti
        self.process_chunk_baseline = process_chunk_baseline
        self.generate_and_estimate_chunk = generate_and_estimate_chunk
    
    def generate_truncated_samples(self, n: int, r: float, 
                                   k: int, key: random.PRNGKey) -> jnp.ndarray:
        """
        Generate k sets of n truncated normal samples.
        
        Parameters
        ----------
        n : int
            Sample size per set
        r : float
            Truncation parameter
        k : int
            Number of sets
        key : random.PRNGKey
            Random key
            
        Returns
        -------
        jnp.ndarray
            Array of shape (n, k)
        """
        # Use rejection sampling for efficiency
        samples = jnp.zeros((n, k))
        
        for i in range(n):
            key_i, key = random.split(key)
            
            # Generate with oversampling
            oversample = 5
            total_needed = k
            
            key_gen, key_i = random.split(key_i)
            candidates = random.normal(key_gen, (oversample * k,))
            mask = (candidates >= -r) & (candidates <= r)
            valid = candidates[mask]
            
            # Take first k samples
            samples = samples.at[i, :].set(valid[:k])
        
        return samples
    
    def run_validation_chunked(self, n_values: List[int], 
                              r_values: List[float],
                              k_total: int,
                              seed: int = 42) -> Dict:
        """
        Run chunked Monte Carlo validation.
        
        Parameters
        ----------
        n_values : list of int
            Sample sizes to test
        r_values : list of float
            Truncation parameters to test
        k_total : int
            Total number of Monte Carlo trials
        seed : int
            Random seed
            
        Returns
        -------
        dict
            Results dictionary with mean estimates and statistics
        """
        key = random.PRNGKey(seed)
        
        # Calculate number of chunks
        n_chunks = int(np.ceil(k_total / self.chunk_size))
        
        print(f"\nRunning validation:")
        print(f"  Sample sizes (n): {n_values}")
        print(f"  Truncation params (r): {r_values}")
        print(f"  Total trials (k): {k_total:,}")
        print(f"  Chunks: {n_chunks}")
        print(f"  Trials per chunk: {self.chunk_size:,}")
        
        results = {
            'n_values': n_values,
            'r_values': r_values,
            'k_total': k_total,
            'mean_opti': np.zeros((len(r_values), len(n_values))),
            'mean_baseline': np.zeros((len(r_values), len(n_values))),
            'std_opti': np.zeros((len(r_values), len(n_values))),
            'std_baseline': np.zeros((len(r_values), len(n_values))),
            'timing': {}
        }
        
        total_start = time.time()
        
        for ri, r in enumerate(r_values):
            for ni, n in enumerate(n_values):
                print(f"\n  Position: n={n}, r={r:.3f}")
                
                # Accumulators for online mean/variance computation
                sum_opti = 0.0
                sum_sq_opti = 0.0
                sum_baseline = 0.0
                sum_sq_baseline = 0.0
                count = 0
                
                chunk_start = time.time()
                
                for chunk_idx in range(n_chunks):
                    # Determine chunk size (last chunk might be smaller)
                    current_chunk_size = min(self.chunk_size, 
                                            k_total - chunk_idx * self.chunk_size)
                    
                    # Generate key for this chunk
                    key, subkey = random.split(key)
                    
                    # Generate samples
                    samples = self.generate_truncated_samples(n, r, 
                                                             current_chunk_size, 
                                                             subkey)
                    
                    # Compute estimates
                    s_opti = self.process_chunk_opti(samples, r,
                                                     self.estimator.nn,
                                                     self.estimator.rr,
                                                     self.estimator.bnr)
                    s_baseline = self.process_chunk_baseline(samples, r)
                    
                    # Update accumulators (Welford's online algorithm)
                    s_opti_np = np.array(s_opti)
                    s_baseline_np = np.array(s_baseline)
                    
                    sum_opti += np.sum(s_opti_np)
                    sum_sq_opti += np.sum(s_opti_np**2)
                    sum_baseline += np.sum(s_baseline_np)
                    sum_sq_baseline += np.sum(s_baseline_np**2)
                    count += current_chunk_size
                    
                    # Progress update
                    if (chunk_idx + 1) % max(1, n_chunks // 10) == 0:
                        progress = 100 * (chunk_idx + 1) / n_chunks
                        print(f"    Chunk {chunk_idx+1}/{n_chunks} ({progress:.1f}%)")
                    
                    # Clear GPU memory periodically
                    if (chunk_idx + 1) % 10 == 0:
                        jax.clear_caches()
                        gc.collect()
                
                chunk_time = time.time() - chunk_start
                
                # Compute final statistics
                mean_opti = sum_opti / count
                mean_baseline = sum_baseline / count
                
                var_opti = (sum_sq_opti / count) - mean_opti**2
                var_baseline = (sum_sq_baseline / count) - mean_baseline**2
                
                std_opti = np.sqrt(max(0, var_opti))
                std_baseline = np.sqrt(max(0, var_baseline))
                
                # Store results
                results['mean_opti'][ri, ni] = mean_opti
                results['mean_baseline'][ri, ni] = mean_baseline
                results['std_opti'][ri, ni] = std_opti
                results['std_baseline'][ri, ni] = std_baseline
                
                print(f"    Mean(opti):     {mean_opti:.6f}")
                print(f"    Mean(baseline): {mean_baseline:.6f}")
                print(f"    Time: {chunk_time:.2f}s")
                
                # Store timing
                results['timing'][f'n{n}_r{r}'] = chunk_time
        
        total_time = time.time() - total_start
        results['total_time'] = total_time
        
        print(f"\n\nTotal validation time: {total_time:.2f}s")
        print(f"Average time per configuration: {total_time/(len(n_values)*len(r_values)):.2f}s")
        
        return results
    
    def save_results(self, results: Dict, filename: str = 'results_std_bnr.npz'):
        """Save results to file."""
        np.savez(filename, **results)
        print(f"\nResults saved to {filename}")
    
    def print_summary(self, results: Dict):
        """Print a summary of results."""
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        
        n_values = results['n_values']
        r_values = results['r_values']
        
        print(f"\nTotal Monte Carlo trials: {results['k_total']:,}")
        print(f"\nTrue sigma: 1.0 (by construction)")
        
        for ri, r in enumerate(r_values):
            print(f"\n\nTruncation parameter r = {r:.1f}")
            print("-" * 70)
            print(f"{'n':>5} {'Mean(Opti)':>12} {'Mean(Base)':>12} "
                  f"{'Bias(Opti)%':>12} {'Bias(Base)%':>12}")
            print("-" * 70)
            
            for ni, n in enumerate(n_values):
                mean_opti = results['mean_opti'][ri, ni]
                mean_base = results['mean_baseline'][ri, ni]
                
                bias_opti = 100 * (mean_opti - 1.0)
                bias_base = 100 * (mean_base - 1.0)
                
                print(f"{n:5d} {mean_opti:12.6f} {mean_base:12.6f} "
                      f"{bias_opti:11.3f}% {bias_base:11.3f}%")


def main():
    """Main validation script."""
    
    # Configuration
    n_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    r_values = [0.3, 0.5, 1.0, 2.0, 5.0]
    
    # Adjust k_total and chunk_size based on your needs
    # For quick test: k_total = 10000
    # For validation: k_total = 1000000 (1e6)
    # For publication: k_total = 100000000 (1e8)
    
    k_total = 200000000  # Start with 1M for testing
    chunk_size = 200000  # Adjust based on GPU memory
    
    print("="*70)
    print("TRUNCATED NORMAL STD ESTIMATION VALIDATION")
    print("="*70)
    
    # Initialize validator
    validator = MonteCarloValidator(use_gpu=True, chunk_size=chunk_size)
    
    # Run validation
    results = validator.run_validation_chunked(
        n_values=n_values,
        r_values=r_values,
        k_total=k_total,
        seed=42
    )
    
    # Print summary
    validator.print_summary(results)
    
    # Save results
    validator.save_results(results, 'results_std_bnr.npz')
    
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
