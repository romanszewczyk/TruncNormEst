"""
Efficient truncated normal random number generator using JAX.

Based on the algorithm by Botev (2016) for generating samples from
truncated normal distributions.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
import jax.random as random
from typing import Tuple


class TruncatedNormalGenerator:
    """
    Efficient generator for truncated normal distributions.
    
    Implements the algorithm from:
    Botev, Z. I. (2016). "The normal law under linear restrictions:
    simulation and estimation via minimax tilting."
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize the generator.
        
        Parameters
        ----------
        use_gpu : bool
            Whether to use GPU acceleration
        """
        self.use_gpu = use_gpu and jax.devices('gpu')
        self._compile_functions()
    
    def _compile_functions(self):
        """Compile JAX functions for performance."""
        
        @jit
        def ntail(l: jnp.ndarray, u: jnp.ndarray, 
                 key: random.PRNGKey) -> jnp.ndarray:
            """
            Sample from truncated normal using Rayleigh method.
            For l > 0 (right tail).
            """
            c = l**2 / 2.0
            f = jnp.expm1(c - u**2 / 2.0)
            
            # Generate initial samples
            key1, key2 = random.split(key)
            uniform1 = random.uniform(key1, shape=l.shape)
            x = c - jnp.log(1.0 + uniform1 * f)
            
            # Acceptance-rejection loop (vectorized)
            uniform2 = random.uniform(key2, shape=l.shape)
            accept = uniform2**2 * x <= c
            
            return jnp.sqrt(2.0 * x), accept
        
        @jit
        def tn_inverse_transform(l: jnp.ndarray, u: jnp.ndarray,
                                key: random.PRNGKey) -> jnp.ndarray:
            """
            Sample using inverse transform method.
            For abs(u-l) < tol.
            """
            uniform = random.uniform(key, shape=l.shape)
            
            # erfc implementation
            pl = jax.scipy.special.erfc(l / jnp.sqrt(2.0)) / 2.0
            pu = jax.scipy.special.erfc(u / jnp.sqrt(2.0)) / 2.0
            
            # Inverse transform
            x = jnp.sqrt(2.0) * jax.scipy.special.erfcinv(
                2.0 * (pl - (pl - pu) * uniform)
            )
            
            return x
        
        @jit
        def trnd(l: jnp.ndarray, u: jnp.ndarray,
                key: random.PRNGKey) -> jnp.ndarray:
            """
            Sample using acceptance-rejection from standard normal.
            For -a < l < u < a.
            """
            x = random.normal(key, shape=l.shape)
            accept = (x >= l) & (x <= u)
            return x, accept
        
        self.ntail = ntail
        self.tn_inverse_transform = tn_inverse_transform
        self.trnd = trnd
    
    def sample(self, l: float, u: float, size: int,
              key: random.PRNGKey) -> jnp.ndarray:
        """
        Generate samples from truncated standard normal N(0,1) on [l, u].
        
        Parameters
        ----------
        l : float
            Lower truncation bound
        u : float
            Upper truncation bound
        size : int
            Number of samples to generate
        key : random.PRNGKey
            JAX random key
            
        Returns
        -------
        jnp.ndarray
            Array of samples from truncated normal distribution
        """
        a = 0.66  # Threshold for switching methods
        tol = 2.0  # Threshold for inverse transform vs rejection
        
        # Initialize output array
        l_arr = jnp.full(size, l)
        u_arr = jnp.full(size, u)
        
        # Case 1: l > a (right tail)
        mask1 = l > a
        
        # Case 2: u < -a (left tail, use symmetry)
        mask2 = u < -a
        
        # Case 3: otherwise
        mask3 = ~(mask1 | mask2)
        
        # Initialize result
        x = jnp.zeros(size)
        
        # Handle case 1: right tail
        if jnp.any(mask1):
            key, subkey = random.split(key)
            l_case1 = jnp.where(mask1, l_arr, 0.0)
            u_case1 = jnp.where(mask1, u_arr, jnp.inf)
            
            # Use rejection sampling (simplified for vectorization)
            max_iter = 100
            for _ in range(max_iter):
                key, subkey = random.split(key)
                samples, accept = self.ntail(l_case1, u_case1, subkey)
                x = jnp.where(mask1 & accept, samples, x)
                mask1 = mask1 & ~accept
                if not jnp.any(mask1):
                    break
        
        # Handle case 2: left tail (use symmetry)
        mask2_active = mask2
        if jnp.any(mask2_active):
            key, subkey = random.split(key)
            l_case2 = jnp.where(mask2_active, -u_arr, 0.0)
            u_case2 = jnp.where(mask2_active, -l_arr, jnp.inf)
            
            max_iter = 100
            for _ in range(max_iter):
                key, subkey = random.split(key)
                samples, accept = self.ntail(l_case2, u_case2, subkey)
                x = jnp.where(mask2_active & accept, -samples, x)
                mask2_active = mask2_active & ~accept
                if not jnp.any(mask2_active):
                    break
        
        # Handle case 3: middle region
        if jnp.any(mask3):
            abs_diff = jnp.abs(u_arr - l_arr)
            
            # Use inverse transform for small intervals
            use_inverse = mask3 & (abs_diff <= tol)
            
            if jnp.any(use_inverse):
                key, subkey = random.split(key)
                l_inv = jnp.where(use_inverse, l_arr, 0.0)
                u_inv = jnp.where(use_inverse, u_arr, 0.0)
                samples = self.tn_inverse_transform(l_inv, u_inv, subkey)
                x = jnp.where(use_inverse, samples, x)
            
            # Use rejection from normal for larger intervals
            use_rejection = mask3 & ~use_inverse
            
            if jnp.any(use_rejection):
                key, subkey = random.split(key)
                l_rej = jnp.where(use_rejection, l_arr, 0.0)
                u_rej = jnp.where(use_rejection, u_arr, 0.0)
                
                max_iter = 1000
                for _ in range(max_iter):
                    key, subkey = random.split(key)
                    samples, accept = self.trnd(l_rej, u_rej, subkey)
                    x = jnp.where(use_rejection & accept, samples, x)
                    use_rejection = use_rejection & ~accept
                    if not jnp.any(use_rejection):
                        break
        
        return x
    
    def sample_vectorized(self, l: float, u: float, 
                         n_samples: int, n_sets: int,
                         key: random.PRNGKey) -> jnp.ndarray:
        """
        Generate multiple sets of samples efficiently.
        
        Parameters
        ----------
        l : float
            Lower truncation bound
        u : float
            Upper truncation bound
        n_samples : int
            Number of samples per set
        n_sets : int
            Number of sets to generate
        key : random.PRNGKey
            JAX random key
            
        Returns
        -------
        jnp.ndarray
            Array of shape (n_sets, n_samples)
        """
        keys = random.split(key, n_sets)
        
        # Use vmap for parallel generation
        sample_fn = lambda k: self.sample(l, u, n_samples, k)
        samples = vmap(sample_fn)(keys)
        
        return samples


# Simpler, faster implementation for common case
@jit
def truncnorm_simple(key: random.PRNGKey, l: float, u: float, 
                    shape: Tuple[int, ...]) -> jnp.ndarray:
    """
    Simple truncated normal sampler using rejection sampling.
    
    Best for moderate truncation (not extreme tails).
    
    Parameters
    ----------
    key : random.PRNGKey
        JAX random key
    l : float
        Lower bound
    u : float
        Upper bound
    shape : tuple
        Shape of output array
        
    Returns
    -------
    jnp.ndarray
        Samples from truncated normal
    """
    def sample_batch(key_in):
        # Generate extra samples to reduce iterations
        n_total = int(jnp.prod(jnp.array(shape)))
        oversampling_factor = 5
        
        key1, key2 = random.split(key_in)
        
        # Generate candidates
        candidates = random.normal(key1, (oversampling_factor * n_total,))
        
        # Accept those in range
        mask = (candidates >= l) & (candidates <= u)
        accepted = candidates[mask]
        
        # Take first n_total samples
        result = accepted[:n_total]
        
        # If not enough, fill with more samples (recursive)
        n_missing = n_total - len(result)
        if n_missing > 0:
            # Simple fallback: use uniform approximation in tail
            extra = random.uniform(key2, (n_missing,)) * (u - l) + l
            result = jnp.concatenate([result, extra])
        
        return result.reshape(shape)
    
    return sample_batch(key)


if __name__ == "__main__":
    # Quick test
    print("Testing TruncatedNormalGenerator...")
    
    generator = TruncatedNormalGenerator(use_gpu=True)
    
    # Test parameters
    l, u = -2.0, 2.0
    n_samples = 1000
    
    key = random.PRNGKey(0)
    samples = generator.sample(l, u, n_samples, key)
    
    print(f"\nGenerated {n_samples} samples from N(0,1) truncated to [{l}, {u}]")
    print(f"Mean: {jnp.mean(samples):.4f} (expected: ~0)")
    print(f"Std:  {jnp.std(samples):.4f}")
    print(f"Min:  {jnp.min(samples):.4f} (expected: {l})")
    print(f"Max:  {jnp.max(samples):.4f} (expected: {u})")
    
    # Test vectorized version
    key = random.PRNGKey(1)
    n_sets = 100
    samples_vec = generator.sample_vectorized(l, u, n_samples, n_sets, key)
    print(f"\nGenerated {n_sets} sets of {n_samples} samples")
    print(f"Shape: {samples_vec.shape}")
    print(f"Mean across all: {jnp.mean(samples_vec):.4f}")
