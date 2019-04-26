close all;
clear all;

rng('default');

N = 64;          % Image is N-by-N pixels
theta = 0:2:178; % projection angles
p = 90;          % Number of rays for each angle

% Assemble the X-ray tomography matrix, the true data, and true image
K = paralleltomo(N, theta, p);

% Generate synthetic data
m_true = phantomgallery('smooth', N);
m_true = m_true(:);

d = K*m_true;

subplot(121);
imagesc(reshape(m_true, N, N));
title('True image');
subplot(122);
imagesc(reshape(d, p, length(theta)));
title('Data (sinograph)');

% Remove possibly 0 rows from K, and d
[K, d] = purge_rows(K, d);

% Rescale K, and d so that the l2 norm of each row is 1
s = sqrt(sum(K.*K, 2) );
K = spdiags(1./s,0)*K;
d = spdiags(1./s,0)*d;

%% A) Reconstruct m using EM. Report D_KL(d || K*m) and ||m - m_true||
K = full(K);
n = size(K,2);
q = size(K,1);
KL_dist = zeros(100,1);
error = zeros(100,1);
S = zeros(n,n);
m = ones(n,1); % Initialize m

colume_sum = sum(K,1);

for k = 1:n
    S(k,k) = colume_sum(k);
end

for j = 1:100
    m = m.*((S\K')*(d./(K*m))); 
    dist_sum = 0;
    
    % KL_distance
    for i = 1:q
        dist_sum = dist_sum + d(i) * log(d(i)./(K(i,:)*m)) + K(i,:)*m - d(i);
    end
    KL_dist(j) = dist_sum;
    error(j) = norm(m_true-m);
end

figure(1);
subplot(121);
imagesc(reshape(m_true, N, N));
title('True image');
subplot(122);
imagesc(reshape(m, N, N));
title('Recon image');

figure(2);
semilogy(KL_dist,'Linewidth', 2);
xlabel('j');ylabel('D_KL(d || K*m)')
title('KL Dist');

figure(3);
semilogy(error,'Linewidth', 2);
xlabel('j');ylabel('||m_true - m_j||')
title('error');

%% B) Consider the noisy data case, i.e d is a realization of a Poisson
% distributed random vector.
% Design an appropriate stopping criterion for EM and reconstruct m
close all;
clear all;

rng('default');

N = 64;          % Image is N-by-N pixels
theta = 0:2:178; % projection angles
p = 90;          % Number of rays for each angle

% Assemble the X-ray tomography matrix, the true data, and true image
K = paralleltomo(N, theta, p);

% Generate synthetic data
m_true = phantomgallery('smooth', N);
m_true = m_true(:);

d = poissrnd(K*m_true);
[K, d] = purge_rows(K, d);
K = full(K);

n = size(K,2);
q = size(K,1);
KL_dist = zeros(100,1);
misfit = zeros(100,1);
S = zeros(n,n);
m = ones(n,1);
%K = full(K);

colume_sum = sum(K,1);
for k = 1:n
    S(k,k) = colume_sum(k);
end

for j = 1:100
    m = m.*((S\K')*(d./(K*m))); 
    KL_dist(j) = nansum((d .* log(d./(K*m)) + K*m - d));
    misfit(j) = norm(K*m - d);
end

figure(1);
subplot(121);
imagesc(reshape(m_true, N, N));
title('True image');
subplot(122);
imagesc(reshape(m, N, N));
title('Recon image');

Dh = d .* log(d./(K*m_true)) + K*m_true - d;
DKL_dh = nansum(Dh);

figure(3);
loglog(KL_dist,'Linewidth', 2);
hold on;
loglog(norm(DKL_dh)*ones(1,100),'r--','Linewidth', 2);
xlabel('j'); ylabel('DKL(d-head||K*m)');
title('Morozov discrepancy criterion');
loglog(5, norm(DKL_dh),'ro','Linewidth',2); % stop at iter = 4
