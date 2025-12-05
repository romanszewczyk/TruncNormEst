clear all
close all
clc

load('optimal_brn_25reps_FINAL.mat');

[rr,nn] = meshgrid(double(n),r_values);

figure(1)
surf(rr,nn,squeeze(mean(brn_matrix,1)));
xlabel('\it{n}');
ylabel('\it{r}');
zlabel('\it{b(n,r)}');


