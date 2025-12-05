clear all
clc

load('optimal_brn_25reps_FINAL.mat');

[nn,rr] = meshgrid(double(n),r_values);

n = double(n);
r = r_values;
rbn_MC_data = squeeze(mean(brn_matrix,1));

figure(1)
surf(nn,rr,rbn_MC_data);
xlabel('\it{n}');
ylabel('\it{r}');
zlabel('\it{b(n,r)}');

figure(2)
surf(nn,rr,2.*sqrt(1./25).*squeeze(std(brn_matrix,1)));
xlabel('\it{n}');
ylabel('\it{r}');
zlabel('\it{std(b(n,r))}');


ni = 2:max(double(n));
ri = 0.2:0.1:max(r);

[nni,rri] = meshgrid(ni,ri);
rbn_MC_datai = interp2(nn,rr,rbn_MC_data,nni,rri,'pchip');

figure (3)
surf(nni,rri,rbn_MC_datai);
xlabel('\it{n}');
ylabel('\it{r}');
zlabel('\it{b(n,r)}');


n = ni;
r = ri;
nn = nni;
rr = rri;

rbn_MC_data = rbn_MC_datai;

save -v7 rbn_MC_data.mat n r rbn_MC_data nn rr
save -text rbn_MC_data.txt n r rbn_MC_data

