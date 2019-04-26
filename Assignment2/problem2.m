close all;
clear all;

rng('default');

N = 64;          % Image is N-by-N pixels
theta = 0:2:178; % projection angles
p = 90;          % Number of rays for each angle


% Assemble the X-ray tomography matrix, the true data, and true image
[K, d, m_true] = paralleltomo(N, theta, p);

subplot(121);
imagesc(reshape(m_true, N, N));
title('True image');
subplot(122);
imagesc(reshape(d, p, length(theta)));
title('Data (sinograph)');

% Remove possibly 0 rows from K and d
[K, d] = purge_rows(K, d);

%% A) Reconstruct m using ART. Report convergence history (i.e. residual and
% error norms at each iteration)
K = full(K);
q = size(K,1);
n = size(K,2);
m = zeros(n,1);
swipe = 0;

di = zeros(q,1);
K_norm = zeros(q,1);
for i = 1:q
    KiT =  K(i,:);
    di(i) = K(i,:)*m_true;
    K_norm(i) = norm(KiT')^2;
end

% ART recon
residual = zeros(1,1000);
error = zeros(1,1000);
while swipe < 1000
    for i = 1:q
        KiT =  K(i,:);
        m = m + ((di(i)-K(i,:)*m)/K_norm(i))*KiT';
    end
    residual(swipe+1) = norm(d-K*m);
    error(swipe+1) = norm(m_true-m);
    swipe = swipe + 1;
end

figure(1);
subplot(121);
imagesc(reshape(m_true, N, N));
title('True image');
subplot(122);
imagesc(reshape(m, N, N));
title('Recon image');
save('Arecon.mat','m');

figure(2);
semilogy(residual,'Linewidth', 2);
xlabel('j');ylabel('||d-Km(j)||')
title('residual')

figure(3);
semilogy(error,'Linewidth', 2);
xlabel('j');ylabel('||m_true - m_j||')
title('error')
save('2Aerror.mat', 'error');

%% B) Reconstruct m using SART. Report convergence history and compare with
%  SART
%K = full(K);
q = size(K,1);
n = size(K,2);
m = zeros(n,1);
D = zeros(q,q);

% Construct D
for i = 1:q
    D(i,i) = 1/(norm(K(i,:)'))^2;
end

% SART recon
swipe = 0;
residual = zeros(1,1000);
error = zeros(1,1000);
w = 1/q;
while swipe < 1000
    m = m + w*K'*D*(d-K*m);
    residual(swipe+1) = norm(d-K*m); % Plot the residual and error
    error(swipe+1) = norm(m_true-m);
    swipe = swipe + 1;
end

figure(1);
subplot(121);
imagesc(reshape(m_true, N, N));
title('True image');
subplot(122);
imagesc(reshape(m, N, N));
title('Recon image');

figure(2);
semilogy(residual,'Linewidth', 2);
xlabel('j');ylabel('||d-Km(j)||')
title('residual')

figure(3);
semilogy(error,'Linewidth', 2);
xlabel('j');ylabel('||m_true - m_j||')
title('error')


%% C) Reconstruct m using SIRT. Report convergence history and compare with
% SIRT
q = size(K,1);
n = size(K,2);
m = zeros(n,1);
D = zeros(q,q);

% Construct D
for i = 1:q
    D(i,i) = 1/(norm(K(i,:)'))^2;
end

% SIRT recon
swipe = 0;
residual = zeros(1,1000);
error = zeros(1,1000);
w = 2/q;
while swipe < 1000
    m = m + w*K'*D*(d-K*m);
    residual(swipe+1) = norm(d-K*m); % Plot the residual and error
    error(swipe+1) = norm(m_true-m);
    swipe = swipe + 1;
end

figure(1);
subplot(121);
imagesc(reshape(m_true, N, N));
title('True image');
subplot(122);
imagesc(reshape(m, N, N));
title('Recon image');

figure(2);
semilogy(residual,'Linewidth', 2);
xlabel('j');ylabel('||d-Km(j)||')
title('residual')

figure(3);
semilogy(error,'Linewidth', 2);
xlabel('j');ylabel('||m_true - m_j||')
title('error')

%% D) Consider the case with noisy data. Reconstruct m using ART, SART,
% SIRT. Report convergence history and discuss what you observed.
noise_level = 0.01; % noise level.
noise_std = noise_level*norm(d,'inf');
dn = d + noise_std*randn(size(d));
q = size(K,1);
n = size(K,2);
K = full(K);
%% ART
m_ART = zeros(n,1);
swipe = 0;
K_norm = zeros(q,1);
for i = 1:q
    KiT =  K(i,:);
    K_norm(i) = norm(KiT')^2;
end

% recon
while swipe < 1000
    for i = 1:q
        KiT =  K(i,:);
        m_ART = m_ART + ((dn(i)-KiT*m_ART)/K_norm(i))*KiT';
    end
    swipe = swipe + 1;
end

figure(1);
subplot(121);
imagesc(reshape(m_true, N, N));
title('True image');
subplot(122);
imagesc(reshape(m, N, N));
title('Recon image');
%% SART
m_SART = zeros(n,1);
for i = 1:q
    D(i,i) = 1/(norm(K(i,:)'))^2;
end

% SART recon
swipe = 0;
%residual = zeros(1,1000);
%error = zeros(1,1000);
w = 1/q;
while swipe < 1000
    m_SART = m_SART + w*K'*D*(dn-K*m_SART);
    %residual(swipe+1) = norm(d-K*m); % Plot the residual and error
    %error(swipe+1) = norm(m_true-m);
    swipe = swipe + 1;
end

figure(1);
subplot(121);
imagesc(reshape(m_true, N, N));
title('True image');
subplot(122);
imagesc(reshape(m, N, N));
title('Recon image');

%% SIRT
m_SART = zeros(n,1);
for i = 1:q
    D(i,i) = 1/(norm(K(i,:)'))^2;
end

% SIRT recon
swipe = 0;
%residual = zeros(1,1000);
%error = zeros(1,1000);
w = 2/q;
while swipe < 1000
    m_SART = m_SART + w*K'*D*(dn-K*m_SART);
    %residual(swipe+1) = norm(d-K*m); % Plot the residual and error
    %error(swipe+1) = norm(m_true-m);
    swipe = swipe + 1;
end

figure(1);
subplot(121);
imagesc(reshape(m_true, N, N));
title('True image');
subplot(122);
imagesc(reshape(m, N, N));
title('Recon image');

%% E) Implement Morozov discrepancy principle as stopping criterion for ART

% ART
K = full(K);
m = zeros(n,1);
K_norm = zeros(q,1);
misfit = zeros(1000,1);
swipe = 0;
for i = 1:q
    KiT =  K(i,:);
    K_norm(i) = norm(KiT')^2;
end

% recon

while swipe < 1000
    for i = 1:q
        KiT =  K(i,:);
        m = m + ((dn(i)-KiT*m)/K_norm(i))*KiT';
    end
    swipe = swipe + 1;
    misfit(swipe) = norm(K*m - dn);
end

figure(1);
loglog(1:1000,misfit,'Linewidth', 2);
hold on;
noise_level = norm(noise_std*randn(size(d)));
loglog(1:1000, noise_level*ones(1000,1), 'r--', 'Linewidth', 2);
xlabel('j'); ylabel('||K*m - d||');
plot(31, misfit(31), 'ro', 'Linewidth', 3);
title('Morozov discrepancy principle')