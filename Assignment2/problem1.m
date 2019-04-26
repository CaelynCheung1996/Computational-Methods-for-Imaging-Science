clear all
close all

rng('default');

% Number of discretization points
N = 128;
gamma = 0.03;
C = 1 / (sqrt(2*pi)*gamma);

K = zeros(N,N);
h = 1/N;
x = linspace(0,1,N)';

% discrete convolution matrix
for l = 1:N
    for k = 1:N
    	K(l,k) = h * C * exp(-(l-k)^2 * h^2 / (2 * gamma^2));
    end
end

% true image
m_true = (x > .2).*(x < .3) + sin(4*pi*x).*(x > 0.5);

% convolved image
d = K * m_true;

%% B)
sigma = 0:0.001:1;
w = [0.8,1,1.5];
j = [10,20,40];

%w = 0.4, observe j
figure(1);
for i = 1:3
    phi = 1-(1-w(2).*sigma).^j(i);
    loglog(sigma,phi,'Linewidth', 2);
    hold on;
end
legend('j = 10','j = 20', 'j = 30');
xlabel('?2'); ylabel('filter function');
title('w = 1');

%%
% j = 20, observe w
figure(2);
for i = 1:3
    phi = 1-(1-w(i).*sigma).^j(2);
    loglog(sigma,phi,'Linewidth', 2);
    hold on;
end
legend('w = 0.8','w = 1','w = 1.5');
xlabel('?2'); ylabel('filter function');
title('j = 20');


%% C) Reconstruct m using the Landweber method. Report convergence history
% (i.e. residual and error norms at each iteration)
m = zeros(N,1);
residual = zeros(1,1000);
error = zeros(1,1000);
w = 2;
count = 0;
while count < 1000
    m = m + w.*K'*(d-K*m);
    residual(count+1) = norm(d-K*m);
    error(count+1) = norm(m_true-m);
    count = count + 1;
end

figure(1);
plot(x,m,x,m_true,'Linewidth', 2);
legend('recon data', 'true data');

figure(2);
semilogy(residual,'Linewidth', 2);
xlabel('j');ylabel('||d-Km(j)||')
title('residual');

figure(3);
semilogy(error,'Linewidth', 2);
xlabel('j');ylabel('||m_true - m_j||')
title('error')


%% D) Add normally distributed noise, and reconstruct m using the Landweber
% method. Report convergence history
dn = d + 0.01 * randn(N,1);
m = zeros(N,1);
residual = zeros(1,1000);
error = zeros(1,1000);
w = 2;
count = 0;
while count < 1000
    m = m + w.*K'*(dn-K*m);
    residual(count+1) = norm(dn-K*m);
    error(count+1) = norm(m_true-m);
    count = count + 1;
end

figure(1);
plot(x,m,x,m_true,'Linewidth', 2); % noisy data
legend('recon data', 'true data');

figure(2);
semilogy(residual,'Linewidth', 2);
xlabel('j');ylabel('||d-Km(j)||')
title('residual');


figure(3);
semilogy(error,'Linewidth', 2);
xlabel('j');ylabel('||m_true - m_j||')
title('error');


%% E) Use L-curve to find out when to stop.
dn = d + 0.01 * randn(N,1);
m = zeros(N,1);
iternum = 1000;
misfit = zeros(iternum,1);
reg = zeros(iternum,1);
w = 1;
count = 0;
while count < iternum
    m = m + w.*K'*(dn-K*m);
    misfit(count+1) = norm(K*m - dn);
    reg(count+1) = norm(m);
    count = count + 1;
end

figure(1);
loglog(misfit, reg, 'Linewidth', 2);
hold on;
xlabel('||K*m - d||'); ylabel('||m||');
loglog(misfit(100), reg(100), 'ro', 'Linewidth', 3); % optimal iteration = 100;
title('L-curve');

% Recon the data using optimal iteration number
dn = d + 0.01 * randn(N,1);
m = zeros(N,1);
w = 1;
count = 0;
while count < 140
    m = m + w.*K'*(dn-K*m);

    count = count + 1;
end

figure(2);
plot(x,m,x,m_true,'Linewidth', 2);
legend('recon data', 'true data');
title('early stop at iteration = 140');
