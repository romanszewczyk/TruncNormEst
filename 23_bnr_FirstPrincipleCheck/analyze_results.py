"""
Analysis and visualization of Monte Carlo validation results.

This script loads results from the validation and creates comprehensive
visualizations and statistical analyses.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Dict, Optional
import os


class ResultsAnalyzer:
    """Analyze and visualize validation results."""
    
    def __init__(self, results_file: str = 'results_std_bnr.npz'):
        """
        Load and initialize analyzer.
        
        Parameters
        ----------
        results_file : str
            Path to results file
        """
        if not os.path.exists(results_file):
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        self.results = dict(np.load(results_file, allow_pickle=True))
        
        print(f"Loaded results from {results_file}")
        print(f"  Total trials: {self.results['k_total']:,}")
        print(f"  n values: {self.results['n_values']}")
        print(f"  r values: {self.results['r_values']}")
    
    def compute_bias_and_rmse(self, true_sigma: float = 1.0) -> Dict:
        """
        Compute bias and RMSE for both estimators.
        
        Parameters
        ----------
        true_sigma : float
            True standard deviation value
            
        Returns
        -------
        dict
            Dictionary with bias and RMSE values
        """
        mean_opti = self.results['mean_opti']
        mean_baseline = self.results['mean_baseline']
        std_opti = self.results['std_opti']
        std_baseline = self.results['std_baseline']
        
        # Bias
        bias_opti = mean_opti - true_sigma
        bias_baseline = mean_baseline - true_sigma
        
        # RMSE = sqrt(bias^2 + variance)
        rmse_opti = np.sqrt(bias_opti**2 + std_opti**2)
        rmse_baseline = np.sqrt(bias_baseline**2 + std_baseline**2)
        
        # Relative bias (percentage)
        rel_bias_opti = 100 * bias_opti / true_sigma
        rel_bias_baseline = 100 * bias_baseline / true_sigma
        
        return {
            'bias_opti': bias_opti,
            'bias_baseline': bias_baseline,
            'rel_bias_opti': rel_bias_opti,
            'rel_bias_baseline': rel_bias_baseline,
            'rmse_opti': rmse_opti,
            'rmse_baseline': rmse_baseline,
        }
    
    def plot_bias_comparison(self, save_path: Optional[str] = None):
        """
        Plot bias comparison between estimators.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save figure
        """
        stats = self.compute_bias_and_rmse()
        n_values = self.results['n_values']
        r_values = self.results['r_values']
        
        n_r = len(r_values)
        fig, axes = plt.subplots(2, (n_r + 1) // 2, figsize=(15, 8))
        axes = axes.flatten()
        
        for ri, r in enumerate(r_values):
            ax = axes[ri]
            
            ax.plot(n_values, stats['rel_bias_opti'][ri, :], 
                   'o-', label='Optimal (b(n,r) + alpha(r))', 
                   linewidth=2, markersize=6)
            ax.plot(n_values, stats['rel_bias_baseline'][ri, :], 
                   's--', label='Baseline (alpha(r) only)', 
                   linewidth=2, markersize=6)
            ax.axhline(y=0, color='k', linestyle=':', alpha=0.3)
            
            ax.set_xlabel('Sample size (n)', fontsize=11)
            ax.set_ylabel('Relative Bias (%)', fontsize=11)
            ax.set_title(f'r = {r:.1f}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        # Remove extra subplots
        for i in range(len(r_values), len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle('Bias Comparison: Optimal vs Baseline Estimator', 
                    fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved bias comparison plot to {save_path}")
        
        plt.show()
    
    def plot_rmse_comparison(self, save_path: Optional[str] = None):
        """Plot RMSE comparison."""
        stats = self.compute_bias_and_rmse()
        n_values = self.results['n_values']
        r_values = self.results['r_values']
        
        n_r = len(r_values)
        fig, axes = plt.subplots(1, n_r, figsize=(5*n_r, 4))
        
        if n_r == 1:
            axes = [axes]
        
        for ri, r in enumerate(r_values):
            ax = axes[ri]
            
            ax.plot(n_values, stats['rmse_opti'][ri, :], 
                   'o-', label='Optimal', linewidth=2, markersize=6)
            ax.plot(n_values, stats['rmse_baseline'][ri, :], 
                   's--', label='Baseline', linewidth=2, markersize=6)
            
            ax.set_xlabel('Sample size (n)', fontsize=11)
            ax.set_ylabel('RMSE', fontsize=11)
            ax.set_title(f'r = {r:.1f}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('RMSE Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved RMSE comparison plot to {save_path}")
        
        plt.show()
    
    def plot_3d_surface(self, save_path: Optional[str] = None):
        """Create 3D surface plot of bias."""
        stats = self.compute_bias_and_rmse()
        n_values = self.results['n_values']
        r_values = self.results['r_values']
        
        # Create meshgrid
        N, R = np.meshgrid(n_values, r_values)
        
        fig = plt.figure(figsize=(15, 6))
        
        # Optimal estimator
        ax1 = fig.add_subplot(121, projection='3d')
        surf1 = ax1.plot_surface(N, R, stats['rel_bias_opti'], 
                                cmap='viridis', alpha=0.8)
        ax1.set_xlabel('Sample size (n)', fontsize=10)
        ax1.set_ylabel('Truncation (r)', fontsize=10)
        ax1.set_zlabel('Relative Bias (%)', fontsize=10)
        ax1.set_title('Optimal Estimator\n(b(n,r) + alpha(r))', 
                     fontsize=12, fontweight='bold')
        fig.colorbar(surf1, ax=ax1, shrink=0.5)
        
        # Baseline estimator
        ax2 = fig.add_subplot(122, projection='3d')
        surf2 = ax2.plot_surface(N, R, stats['rel_bias_baseline'], 
                                cmap='plasma', alpha=0.8)
        ax2.set_xlabel('Sample size (n)', fontsize=10)
        ax2.set_ylabel('Truncation (r)', fontsize=10)
        ax2.set_zlabel('Relative Bias (%)', fontsize=10)
        ax2.set_title('Baseline Estimator\n(alpha(r) only)', 
                     fontsize=12, fontweight='bold')
        fig.colorbar(surf2, ax=ax2, shrink=0.5)
        
        plt.suptitle('Bias Surface Comparison', fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved 3D surface plot to {save_path}")
        
        plt.show()
    
    def plot_heatmap(self, save_path: Optional[str] = None):
        """Create heatmap of relative bias improvement."""
        stats = self.compute_bias_and_rmse()
        n_values = self.results['n_values']
        r_values = self.results['r_values']
        
        # Compute improvement (reduction in absolute bias)
        abs_bias_opti = np.abs(stats['rel_bias_opti'])
        abs_bias_baseline = np.abs(stats['rel_bias_baseline'])
        improvement = 100 * (abs_bias_baseline - abs_bias_opti) / abs_bias_baseline
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        im = ax.imshow(improvement, aspect='auto', cmap='RdYlGn', 
                      vmin=0, vmax=100)
        
        # Set ticks
        ax.set_xticks(np.arange(len(n_values)))
        ax.set_yticks(np.arange(len(r_values)))
        ax.set_xticklabels(n_values)
        ax.set_yticklabels([f'{r:.1f}' for r in r_values])
        
        ax.set_xlabel('Sample size (n)', fontsize=12)
        ax.set_ylabel('Truncation parameter (r)', fontsize=12)
        ax.set_title('Bias Reduction: Optimal vs Baseline (%)', 
                    fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Improvement (%)', fontsize=11)
        
        # Add text annotations
        for i in range(len(r_values)):
            for j in range(len(n_values)):
                text = ax.text(j, i, f'{improvement[i, j]:.1f}',
                             ha="center", va="center", color="black", 
                             fontsize=7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved heatmap to {save_path}")
        
        plt.show()
    
    def print_detailed_statistics(self):
        """Print detailed statistical analysis."""
        stats = self.compute_bias_and_rmse()
        n_values = self.results['n_values']
        r_values = self.results['r_values']
        
        print("\n" + "="*80)
        print("DETAILED STATISTICAL ANALYSIS")
        print("="*80)
        
        print(f"\nTotal Monte Carlo trials: {self.results['k_total']:,}")
        print(f"True sigma: 1.0")
        
        for ri, r in enumerate(r_values):
            print(f"\n\n{'-'*80}")
            print(f"TRUNCATION PARAMETER: r = {r:.1f}")
            print(f"{'-'*80}")
            
            print(f"\n{'n':>5} | {'Mean(Opt)':>10} | {'Mean(Base)':>10} | "
                  f"{'Bias(Opt)%':>11} | {'Bias(Base)%':>12} | {'Improv%':>9}")
            print("-" * 80)
            
            for ni, n in enumerate(n_values):
                mean_opti = self.results['mean_opti'][ri, ni]
                mean_base = self.results['mean_baseline'][ri, ni]
                bias_opti = stats['rel_bias_opti'][ri, ni]
                bias_base = stats['rel_bias_baseline'][ri, ni]
                
                # Calculate improvement
                abs_bias_opti = abs(bias_opti)
                abs_bias_base = abs(bias_base)
                improvement = 100 * (abs_bias_base - abs_bias_opti) / abs_bias_base
                
                print(f"{n:5d} | {mean_opti:10.6f} | {mean_base:10.6f} | "
                      f"{bias_opti:10.3f}% | {bias_base:11.3f}% | "
                      f"{improvement:8.1f}%")
            
            # Summary statistics for this r
            print("\nSummary for r = {:.1f}:".format(r))
            avg_bias_opti = np.mean(np.abs(stats['rel_bias_opti'][ri, :]))
            avg_bias_base = np.mean(np.abs(stats['rel_bias_baseline'][ri, :]))
            avg_improvement = 100 * (avg_bias_base - avg_bias_opti) / avg_bias_base
            
            print(f"  Average |bias| - Optimal:  {avg_bias_opti:.3f}%")
            print(f"  Average |bias| - Baseline: {avg_bias_base:.3f}%")
            print(f"  Average improvement:       {avg_improvement:.1f}%")
    
    def create_full_report(self, output_dir: str = './validation_report'):
        """Generate complete validation report with all plots."""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nGenerating full validation report in {output_dir}/")
        
        # Print statistics
        self.print_detailed_statistics()
        
        # Generate all plots
        print("\nGenerating plots...")
        self.plot_bias_comparison(f'{output_dir}/bias_comparison.png')
        self.plot_rmse_comparison(f'{output_dir}/rmse_comparison.png')
        self.plot_3d_surface(f'{output_dir}/bias_surface_3d.png')
        self.plot_heatmap(f'{output_dir}/improvement_heatmap.png')
        
        print(f"\nReport generation complete!")
        print(f"All files saved to {output_dir}/")


def main():
    """Main analysis script."""
    
    print("="*80)
    print("VALIDATION RESULTS ANALYSIS")
    print("="*80)
    
    # Load and analyze results
    analyzer = ResultsAnalyzer('results_std_bnr.npz')
    
    # Generate full report
    analyzer.create_full_report('./validation_report')


if __name__ == "__main__":
    main()
