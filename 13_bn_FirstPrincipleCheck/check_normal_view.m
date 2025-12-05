clear all
clc

k = 5e6;      % number of tries
nT = 2:20;     % check points

sqr = @(x) x.*x;

targetF = @(x) sum(sqr(x-1));

load('Res_check_normal.mat');

printf('Parameters quality:\n');
printf('original:                        %1.5e\n', targetF(stdT)  );
printf('n-1:                             %1.5e\n', targetF(std1T) );
printf('n-1.5:                           %1.5e\n', targetF(std15T));
printf('n-1.5+1/(8*(n-1)):               %1.5e\n', targetF(stdVT) );
printf('n-(1.521*n-1.6415)/(1.014*n-1)): %1.5e\n', targetF(stdo1T) );
printf('n-b(n):                          %1.5e\n', targetF(stdo2T) );
printf('\n\n');


figure (1)
plot (...
      nT, std1T, '-or', ...
      nT, std15T,'-om', ...
      nT, stdVT, '-ob', ...
      nT, stdo1T,'-sk', ...
      nT, stdo2T,'-xk');
xlim([2,max(nT)]);
% ylim([0 1.1]);
grid;
xlabel('\it{n}');
ylabel('\it{E[s]}');
legend({'b(n) = 1','b(n) = 1.5','b(n) = 1.5+1/(8(n-1))','LFF b(n)','tabulated b(n)'},'Location','northeast');

figure (2)
semilogy(...
         nT, abs(std1T-1), '-or', ...
         nT, abs(std15T-1),'-om', ...
         nT, abs(stdVT-1), '-ob', ...
         nT, abs(stdo1T-1),'-sk', ...
         nT, abs(stdo2T-1),'-xk');
xlim([2,max(nT)]);
% ylim([0 1.1]);
grid;
xlabel('\it{n}');
ylabel('\it{|1-E[s]|}');
legend({'b(n) = 1','b(n) = 1.5','b(n) = 1.5+1/(8(n-1))','LFF b(n)','tabulated b(n)'},'Location','northeast');



