clear all
close all

load('Results_v14_improved.mat');

[nn,rr] = meshgrid(n,r_values);

figure(1)
subplot(1,2,1)
surf(nn,rr,rbn_mean);
zlim([1 1.5]);
subplot(1,2,2)
surf(nn,rr,rbn_calc);
zlim([1 1.5]);

sum(sum((rbn_mean-rbn_calc).^2))

r_control = 35;

P_vals = P_final; %params_s1(1:r_control);
A_vals = A_final; %params_s1(r_control+1:2*r_control);
B_vals = B_final; %params_s1(2*r_control+1:3*r_control);
Q_vals = Q_final; %params_s1(3*r_control+1:4*r_control);
C_vals = C_final; %params_s1(4*r_control+1:5*r_control);

figure;
subplot(2,1,1);
plot (r_control_points,P_vals,'-ok',r_control_points,Q_vals,'-or'); grid on;
xlabel('\it{r}');
ylabel('\it{P(r), Q(r)}');
subplot(2,1,2);
plot (r_control_points,P_vals./Q_vals,'-ob'); grid on;
xlabel('\it{r}');
ylabel('\it{P(r)/Q(r)}');

figure;
subplot(3,1,1);
plot (r_control_points,A_vals,'-ok',r_control_points,C_vals,'-or'); grid on;
xlabel('\it{r}');
ylabel('\it{A(r), C(r)}');
subplot(3,1,2);
plot (r_control_points,A_vals./C_vals,'-ob'); grid on;
xlabel('\it{r}');
ylabel('\it{A(r)/C(r)}');
subplot(3,1,3); plot (r_control_points,B_vals,'-ok'); grid on;
xlabel('\it{r}');
ylabel('\it{B(r)}');

