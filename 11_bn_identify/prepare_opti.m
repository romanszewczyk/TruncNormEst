clear all
clc

load('Results_ident_bn.mat');

fprintf('n = [ ...\n');
for i =1:numel(n)
  if i<numel(n)
    fprintf('%i, ',n(i));
  else
    fprintf('%i',n(i));
  end

  if i./10==round(i./10)
    fprintf('... \n');
  end

end
%
fprintf(']; \n\n');

fprintf('bn_mean = [ ...\n');
for i =1:numel(n)
  if i<numel(n)
    fprintf('%1.5f, ',bn_mean(i));
  else
    fprintf('%1.5f',bn_mean(i));
  end

  if i./10==round(i./10)
    fprintf('... \n');
  end

end
%
fprintf(']; \n\n');

fprintf('bn_std = [ ...\n');
for i =1:numel(n)
  if i<numel(n)
    fprintf('%1.5f, ',bn_std(i));
  else
    fprintf('%1.5f',bn_std(i));
  end

  if i./10==round(i./10)
    fprintf('... \n');
  end

end
%
fprintf(']; \n\n');



