clear all
clc

function s = std15(x)
  % x is n x m
  [n, m] = size(x);

  % column means: 1 x m
  mu = sum(x, 1) ./ n;

  % center data: subtract column means
  xc = x - ones(n, 1) * mu;

  % unbiased column variances: 1 x m
  s2 = sum(xc.^2, 1) ./ (n - 1.5);

  % standard deviations: 1 x m
  s = sqrt(s2);
end
%


load('optimal_bn_500times.mat');
n = double(n);


figure (1)
plot(n,bn_mean,'-ok');
xlabel('\it{n}');
ylabel('\it{b(n)}');
grid;

figure (2)
plot(n,bn_mean,'-ok');
xlabel('\it{n}');
ylabel('\it{b(n)}');
xlim([0,30]);
grid;

%figure (3)
%plot(n,std(bn_all),'ok',n,std15(bn_all),'or',n,bn_std,'ob');
%xlabel('\it{n}');
%ylabel('\it{std(b(n)) - black, std15(b(n)) - red }');
%grid;

figure (3)
%plot(n,std(bn_all)./sqrt(size(bn_all,1)),'ok',n,std15(bn_all)./sqrt(size(bn_all,1)),'or');
plot(n,std15(bn_all)./sqrt(size(bn_all,1)),'or');
xlabel('\it{n}');
ylabel('\it{std_m(b(n)) - red, }');
grid;

%fprintf('\n');
%
%for i=1:numel(n);
%  %fprintf('n = %1.1f, b(n)=%1.4f +/- %1.4f \n',n(i),bn_mean(i),2.*bn_std(i));
%  fprintf('n = %1.1f, b(n)=%1.4f \n',n(i),bn_mean(i));
%end
%%
%
%fprintf('\n');


save -v7 Results_ident_bn.mat n bn_mean bn_std

