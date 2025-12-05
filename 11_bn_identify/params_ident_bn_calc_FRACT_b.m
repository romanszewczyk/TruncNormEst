clear all
pkg load optim

load ('Results_ident_bn.mat');

fprintf('Parameters of rational model b(n) = (An + B)/(Cn - 1) ... ');

a_resT = [];
objT = [];

k = 100;

for j = 1:k

    bn_mean_ = 0.*bn_mean;

    for i = 1:numel(bn_mean)
       bn_mean_(i) = bn_mean(i) +randn(1,1).*bn_std(i);
    end
    %

    target = @(a) sum(((a(1).*n+a(2))./(a(3).*n-1) - bn_mean_).^2);

    a0 = [1.567181 -1.650868 1.044165];

    ctl.XVmin = [0.9.*a0(1) 1.1.*a0(2) 1.1.*a0(3)];
    ctl.XVmax = [1.1.*a0(1) 0.9.*a0(2) 0.9.*a0(3)];
    ctl.constr= 0;
    ctl.NP    = 200;
    ctl.refresh  = 0;
    ctl.tol   = 1.e-9;

    [a_res, obj_value, nfeval, convergence] = de_min(target,ctl);

    a_resT = [a_resT; a_res];
    objT = [objT; obj_value];

end
%
fprintf('done. \n');

fprintf('A = %1.6f +/- %1.6f \n',mean(a_resT(:,1)),2.*(1./sqrt(k)).*std(a_resT(:,1)));
fprintf('B = %1.6f +/- %1.6f \n',mean(a_resT(:,2)),2.*(1./sqrt(k)).*std(a_resT(:,2)));
fprintf('C = %1.6f +/- %1.6f \n',mean(a_resT(:,3)),2.*(1./sqrt(k)).*std(a_resT(:,3)));
fprintf('target = %e +/- %1.6f \n',mean(objT),2.*(1./sqrt(k)).*std(objT));
fprintf('k = %i \n\n',k);

for i = 1:20
  fprintf('n = %i, b(n) = %1.6f, b_aprox(n) = %1.6f \n',n(i),bn_mean(i), ...
           (mean(a_resT(:,1)).*n(i)+mean(a_resT(:,2)))./mean((a_resT(:,3)).*n(i)-1) ...
           );
end
%
fprintf('\n\n');

save -v7 Results_rational_model.mat a_resT objT k



