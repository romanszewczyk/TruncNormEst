"""
Quick test script to verify installation and GPU setup.

Run this first to ensure everything is working correctly.
"""

import sys

def check_imports():
    """Check if all required packages are installed."""
    print("Checking package imports...")
    
    packages = {
        'numpy': None,
        'jax': None,
        'scipy': None,
        'matplotlib': None,
        'seaborn': None
    }
    
    for package in packages:
        try:
            module = __import__(package)
            packages[package] = getattr(module, '__version__', 'unknown')
            print(f"  {package:12s} OK (version {packages[package]})")
        except ImportError as e:
            print(f"  {package:12s} MISSING - {e}")
            return False
    
    return True


def check_jax_gpu():
    """Check JAX GPU configuration."""
    print("\nChecking JAX GPU setup...")
    
    try:
        import jax
        
        # Check default backend
        backend = jax.default_backend()
        print(f"  Default backend: {backend}")
        
        # List devices
        devices = jax.devices()
        print(f"  Total devices: {len(devices)}")
        
        for i, device in enumerate(devices):
            print(f"    Device {i}: {device}")
        
        # Check for GPU specifically
        gpu_devices = jax.devices('gpu')
        if gpu_devices:
            print(f"\n  GPU devices found: {len(gpu_devices)}")
            for gpu in gpu_devices:
                print(f"    {gpu}")
            return True
        else:
            print("\n  WARNING: No GPU devices found!")
            print("  JAX will use CPU (much slower)")
            print("  To enable GPU:")
            print("    pip install jax[cuda12]  # For CUDA 12")
            print("    pip install jax[cuda11]  # For CUDA 11")
            return False
            
    except Exception as e:
        print(f"  ERROR checking JAX: {e}")
        return False


def test_basic_functionality():
    """Test basic estimator functionality."""
    print("\nTesting basic functionality...")
    
    try:
        from truncated_normal_estimator import TruncatedNormalEstimator
        import jax.numpy as jnp
        
        # Create estimator
        estimator = TruncatedNormalEstimator(use_gpu=True)
        print("  Estimator initialized OK")
        
        # Test data
        x = jnp.array([0.5, -0.3, 0.8, -0.2, 0.1, 0.4, -0.5, 0.6, -0.1, 0.3])
        r = 2.0
        
        # Compute estimates
        s_opti = estimator.std_bnr_opti(x, r)
        s_baseline = estimator.std_bnr(x, r)
        
        print(f"  Test computation successful:")
        print(f"    Optimal:  {s_opti:.6f}")
        print(f"    Baseline: {s_baseline:.6f}")
        print(f"    Ratio:    {s_opti/s_baseline:.6f}")
        
        # Sanity check
        if 0.3 < s_opti < 2.0 and 0.3 < s_baseline < 2.0:
            print("  Results look reasonable!")
            return True
        else:
            print("  WARNING: Results may be incorrect")
            return False
            
    except Exception as e:
        print(f"  ERROR during test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_random_generation():
    """Test truncated normal generator."""
    print("\nTesting random number generation...")
    
    try:
        from truncated_normal_generator import TruncatedNormalGenerator
        import jax.random as random
        import jax.numpy as jnp
        
        generator = TruncatedNormalGenerator(use_gpu=True)
        print("  Generator initialized OK")
        
        # Generate samples
        key = random.PRNGKey(42)
        l, u = -2.0, 2.0
        n_samples = 1000
        
        samples = generator.sample(l, u, n_samples, key)
        
        # Check properties
        mean = jnp.mean(samples)
        std = jnp.std(samples)
        min_val = jnp.min(samples)
        max_val = jnp.max(samples)
        
        print(f"  Generated {n_samples} samples:")
        print(f"    Mean: {mean:.4f} (expected ~0)")
        print(f"    Std:  {std:.4f}")
        print(f"    Min:  {min_val:.4f} (expected ~{l})")
        print(f"    Max:  {max_val:.4f} (expected ~{u})")
        
        # Sanity checks
        if l <= min_val and max_val <= u and -0.2 < mean < 0.2:
            print("  Generation test passed!")
            return True
        else:
            print("  WARNING: Generation results unexpected")
            return False
            
    except Exception as e:
        print(f"  ERROR during generation test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mini_validation():
    """Run a tiny validation to ensure everything works."""
    print("\nRunning mini validation (this may take 10-30 seconds)...")
    
    try:
        from monte_carlo_validation import MonteCarloValidator
        
        # Very small test
        validator = MonteCarloValidator(use_gpu=True, chunk_size=100)
        
        results = validator.run_validation_chunked(
            n_values=[5, 10],
            r_values=[1.0, 2.0],
            k_total=1000,  # Only 1000 trials for quick test
            seed=42
        )
        
        print("\n  Mini validation complete!")
        print("  Sample result (n=5, r=1.0):")
        print(f"    Optimal:  {results['mean_opti'][0, 0]:.6f}")
        print(f"    Baseline: {results['mean_baseline'][0, 0]:.6f}")
        
        return True
        
    except Exception as e:
        print(f"  ERROR during validation: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("TRUNCATED NORMAL ESTIMATOR - INSTALLATION TEST")
    print("="*70)
    
    tests = [
        ("Package Imports", check_imports),
        ("JAX GPU Setup", check_jax_gpu),
        ("Basic Functionality", test_basic_functionality),
        ("Random Generation", test_random_generation),
        ("Mini Validation", test_mini_validation),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{'='*70}")
        print(f"TEST: {name}")
        print(f"{'='*70}")
        
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\nFATAL ERROR in {name}: {e}")
            results.append((name, False))
        
        print()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for name, success in results:
        status = "PASS" if success else "FAIL"
        symbol = "OK" if success else "!!"
        print(f"  [{symbol}] {name:30s} {status}")
    
    all_passed = all(success for _, success in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("ALL TESTS PASSED!")
        print("\nYou can now run:")
        print("  python monte_carlo_validation.py")
        print("\nFor full validation with default settings.")
    else:
        print("SOME TESTS FAILED")
        print("\nPlease fix the issues above before proceeding.")
        print("Check the README.md for troubleshooting help.")
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
