% analyze_results_octave.m
% OCTAVE script for visualizing truncated normal validation results
% Loads results from Python NumPy files and creates comprehensive plots

clear all;
close all;
clc;

fprintf('Truncated Normal Validation - OCTAVE Visualization\n');
fprintf('===================================================\n\n');

% Check if required packages are available
if ~exist('OCTAVE_VERSION', 'builtin')
    error('This script requires OCTAVE');
end

% Load results from NumPy file
% Note: You need to convert .npz to .mat first using Python:
%   import numpy as np
%   import scipy.io as sio
%   data = np.load('results_std_bnr.npz')
%   sio.savemat('results_std_bnr.mat', {k: data[k] for k in data.files})

results_file = 'results_std_bnr.mat';

if ~exist(results_file, 'file')
    fprintf('ERROR: Results file not found: %s\n', results_file);
    fprintf('\nPlease convert NumPy results to MAT format first:\n');
    fprintf('  Python code:\n');
    fprintf('    import numpy as np\n');
    fprintf('    import scipy.io as sio\n');
    fprintf('    data = np.load(''results_std_bnr.npz'')\n');
    fprintf('    sio.savemat(''results_std_bnr.mat'', {k: data[k] for k in data.files})\n');
    return;
end

fprintf('Loading results from %s...\n', results_file);
results = load(results_file);

% Extract data
mean_opti = results.mean_opti;
mean_baseline = results.mean_baseline;
n_values = results.n_values(:)';
r_values = results.r_values(:)';
k_total = results.k_total;

fprintf('Loaded results:\n');
fprintf('  k_total = %d\n', k_total);
fprintf('  n values: %d elements\n', length(n_values));
fprintf('  r values: %d elements\n', length(r_values));
fprintf('\n');

% Compute statistics
true_sigma = 1.0;
bias_opti = mean_opti - true_sigma;
bias_baseline = mean_baseline - true_sigma;

rel_bias_opti = 100 * bias_opti / true_sigma;
rel_bias_baseline = 100 * bias_baseline / true_sigma;

abs_bias_opti = abs(rel_bias_opti);
abs_bias_baseline = abs(rel_bias_baseline);
improvement = 100 * (abs_bias_baseline - abs_bias_opti) ./ abs_bias_baseline;

%% Plot 1: Bias Comparison (Multiple Subplots)
fprintf('Creating bias comparison plots...\n');

n_r = length(r_values);
n_cols = ceil(sqrt(n_r));
n_rows = ceil(n_r / n_cols);

figure(1);
set(gcf, 'Position', [100, 100, 1400, 900]);

for ri = 1:4
    subplot(2, 2, ri);

    plot(n_values, 1+rel_bias_opti(ri, :)./100, '-or', 'LineWidth', 2, 'MarkerSize', 6);
    hold on;
    plot(n_values, 1+rel_bias_baseline(ri, :)./100, '-sk', 'LineWidth', 2, 'MarkerSize', 6);
    % plot([min(n_values), max(n_values)], [0, 0], 'k-', 'LineWidth', 1);
    hold off;

    xlabel('\it{n}', 'FontSize', 18);
    ylabel('\it{\sigma reproduction by estimator}', 'FontSize', 18);
    title(sprintf('r = %.1f', r_values(ri)), 'FontSize', 15, 'FontWeight', 'bold');
    %legend('Optimal (b(n,r) + alpha(r))', 'Baseline (alpha(r) only)', ...
    %       'Location', 'best', 'FontSize', 9);
    grid on;
end
%

%sgtitle('Bias Comparison: Optimal vs Baseline Estimator', ...
%        'FontSize', 14, 'FontWeight', 'bold');

print('-dpng', '-r300', 'bias_comparison_octave.png');
fprintf('  Saved: bias_comparison_octave.png\n');

%% Plot 2: 3D Surface Plots
fprintf('Creating 3D surface plots...\n');

[N, R] = meshgrid(n_values, r_values);

figure(2);
set(gcf, 'Position', [100, 100, 1400, 600]);

% Optimal estimator
subplot(1, 2, 1);
surf(N, R, rel_bias_opti);
xlabel('\it{n}', 'FontSize', 18);
ylabel('\it{r}', 'FontSize', 18);
zlabel('\it{Relative bias of the estimator (%)}', 'FontSize', 18);
%title('Optimal Estimator (b(n,r) + alpha(r))', 'FontSize', 12, 'FontWeight', 'bold');
%colorbar;
%shading interp;
view(-30, 30);

% Baseline estimator
subplot(1, 2, 2);
surf(N, R, rel_bias_baseline);
xlabel('\it{n}', 'FontSize', 18);
ylabel('\it{r}', 'FontSize', 18);
zlabel('\it{Relative bias of the estimator (%)}', 'FontSize', 18);
%title('Baseline Estimator (alpha(r) only)', 'FontSize', 12, 'FontWeight', 'bold');
%colorbar;
%shading interp;
view(-40, 30);

sgtitle('Bias Surface Comparison', 'FontSize', 14, 'FontWeight', 'bold');

print('-dpng', '-r300', 'bias_surface_3d_octave.png');
fprintf('  Saved: bias_surface_3d_octave.png\n');

%% Plot 3: Improvement Heatmap
fprintf('Creating improvement heatmap...\n');

figure(3);
set(gcf, 'Position', [100, 100, 1200, 600]);

imagesc(improvement);
set(gca, 'XTick', 1:length(n_values));
set(gca, 'XTickLabel', arrayfun(@num2str, n_values, 'UniformOutput', false));
set(gca, 'YTick', 1:length(r_values));
set(gca, 'YTickLabel', arrayfun(@(x) sprintf('%.1f', x), r_values, 'UniformOutput', false));

xlabel('Sample size (n)', 'FontSize', 12);
ylabel('Truncation parameter (r)', 'FontSize', 12);
title('Bias Reduction: Optimal vs Baseline (%)', 'FontSize', 14, 'FontWeight', 'bold');
colorbar;
colormap(jet);

% Add text annotations
for i = 1:length(r_values)
    for j = 1:length(n_values)
        text(j, i, sprintf('%.1f', improvement(i, j)), ...
             'HorizontalAlignment', 'center', ...
             'VerticalAlignment', 'middle', ...
             'FontSize', 7, ...
             'Color', 'black');
    end
end

print('-dpng', '-r300', 'improvement_heatmap_octave.png');
fprintf('  Saved: improvement_heatmap_octave.png\n');

%% Plot 4: Absolute Bias Comparison
fprintf('Creating absolute bias comparison...\n');

figure(4);
set(gcf, 'Position', [100, 100, 1200, 800]);

for ri = 1:min(4, n_r)
    subplot(2, 2, ri);

    semilogy(n_values, abs_bias_opti(ri, :)./100, '-or', 'LineWidth', 2, 'MarkerSize', 6);
    hold on;
    semilogy(n_values, abs_bias_baseline(ri, :)./100, '-sk', 'LineWidth', 2, 'MarkerSize', 6);
    hold off;

    xlabel('\it{n}', 'FontSize', 18);
    ylabel('Relative bias of estimator', 'FontSize', 18);
    title(sprintf('r = %.1f', r_values(ri)), 'FontSize', 15, 'FontWeight', 'bold');
    %legend('Optimal', 'Baseline', 'Location', 'best', 'FontSize', 9);
    grid on;
end

%sgtitle('Absolute Bias Comparison (Log Scale)', 'FontSize', 14, 'FontWeight', 'bold');

print('-dpng', '-r300', 'absolute_bias_octave.png');
fprintf('  Saved: absolute_bias_octave.png\n');

%% Print Summary Statistics
fprintf('\n');
fprintf('===================================================\n');
fprintf('SUMMARY STATISTICS\n');
fprintf('===================================================\n');
fprintf('Total Monte Carlo trials: %d\n', k_total);
fprintf('True sigma: %.1f\n\n', true_sigma);

for ri = 1:length(r_values)
    fprintf('Truncation parameter r = %.1f\n', r_values(ri));
    fprintf('-----------------------------------------------------\n');
    fprintf('%5s | %12s | %12s | %12s | %12s\n', ...
            'n', 'Mean(Opti)', 'Mean(Base)', 'Bias(Opti)%', 'Bias(Base)%');
    fprintf('-----------------------------------------------------\n');

    for ni = 1:length(n_values)
        fprintf('%5d | %12.6f | %12.6f | %11.3f%% | %11.3f%%\n', ...
                n_values(ni), ...
                mean_opti(ri, ni), ...
                mean_baseline(ri, ni), ...
                rel_bias_opti(ri, ni), ...
                rel_bias_baseline(ri, ni));
    end

    avg_bias_opti = mean(abs_bias_opti(ri, :));
    avg_bias_baseline = mean(abs_bias_baseline(ri, :));
    avg_improvement = mean(improvement(ri, :));

    fprintf('\nSummary for r = %.1f:\n', r_values(ri));
    fprintf('  Average |bias| - Optimal:  %.3f%%\n', avg_bias_opti);
    fprintf('  Average |bias| - Baseline: %.3f%%\n', avg_bias_baseline);
    fprintf('  Average improvement:       %.1f%%\n\n', avg_improvement);
end

fprintf('===================================================\n');
fprintf('All plots saved successfully!\n');
fprintf('===================================================\n');
