clear all
clc

k = 2e8;       % number of tries
nT = 2:20;     % check points

sqr = @(x) x.*x;
targetF = @(x) sum(sqr(x-1));

% --- Definitions for denominators (Scalar operations now) ---
% We define the denominator functions to apply to the pre-calculated sum-squares
calc_d1   = @(n) sqrt(n - 1);
calc_d15  = @(n) sqrt(n - 1.5);
calc_dV   = @(n) sqrt(n - 1.5 + 1./(8.*(n - 1.5)));
calc_do1  = @(n) sqrt(n - (1.521.*n - 1.6415)./(1.014.*n - 1));

% --- Pre-allocate arrays (Prevents memory reallocation in loop) ---
num_loops = numel(nT);
stdT   = zeros(num_loops, 1);
std1T  = zeros(num_loops, 1);
std15T = zeros(num_loops, 1);
stdVT  = zeros(num_loops, 1);
stdo1T = zeros(num_loops, 1);
stdo2T = zeros(num_loops, 1);

fprintf('Modelling started for k= %1.2e repetitions:\n', k);
fprintf([repmat(' ', 1, num_loops) '*\n']); % fast print header

start_time = time();

% Check if std_opti exists, otherwise define a dummy to prevent crash
if ~exist('std_opti', 'file') && ~exist('std_opti', 'var')
    warning('std_opti function not found. Using standard std for stdo2T to prevent crash.');
    std_opti = @(x) std(x);
end

for ni = 1:num_loops
    n = nT(ni);

    % 1. Generate Data
    T = randn(n, k);

    % 2. Core Optimization: Calculate Root Sum of Squares (RSS) ONCE
    % This replaces the repeated calls to mean, subtraction, square, and sum.
    % We center the data manually once.
    % Note: std(T) = sqrt(sum((T-mean(T)).^2) / (n-1))

    % Center the data (fast broadcasting)
    CenterT = T - mean(T);

    % Calculate the "Root Sum of Squared Errors" vector (1 x k)
    % We compute sqrt(sum) here so we can average it immediately.
    % Averaging the sqrt values is required because E[sqrt(x)] != sqrt(E[x])
    RSS_vec = sqrt(sum(CenterT .* CenterT));

    % Get the mean of the unscaled estimators
    mean_RSS = mean(RSS_vec);

    % 3. Apply corrections to the scalar result (Instantaneous)
    stdT(ni)   = mean_RSS / calc_d1(n);  % Standard n-1
    std1T(ni)  = stdT(ni);               % Identical to n-1
    std15T(ni) = mean_RSS / calc_d15(n);
    stdVT(ni)  = mean_RSS / calc_dV(n);
    stdo1T(ni) = mean_RSS / calc_do1(n);

    % std_opti is external/complex, so we must pass T explicitly
    % (We pass T, not CenterT, in case the function logic requires raw data)
    stdo2T(ni) = mean(std_opti(T));

    fprintf('.');
end

elapsed_time = time() - start_time;
printf('\nModelling done in %.2f seconds\n\n', elapsed_time);

printf('Parameters quality:\n');
printf('original:                        %1.5e\n', targetF(stdT)  );
printf('n-1:                             %1.5e\n', targetF(std1T) );
printf('n-1.5:                           %1.5e\n', targetF(std15T));
printf('n-1.5+1/(8*(n-1)):               %1.5e\n', targetF(stdVT) );
printf('n-(1.521*n-1.6415)/(1.014*n-1)): %1.5e\n', targetF(stdo1T) );
printf('n-b(n):                          %1.5e\n', targetF(stdo2T) );
printf('\n\n');

save -v7 Res_check_normal.mat stdT std1T std15T stdVT stdo1T stdo2T
