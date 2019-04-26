clear all
close all

% Number of discretization points
N = 200;
% gamma = 0.03;
% C = 1 / (sqrt(2*pi)*gamma);
C = 0.2;

K = zeros(N,N);
h = 1/N;
x = linspace(0,1,N)';

% discrete convolution matrix
for l = 1:N
    for k = 1:N
    	% K(l,k) = h * C * exp(-(l-k)^2 * h^2 / (2 * gamma^2));
        K(l,k) = h * (1/C^2).* max(0,C-abs((l-k)*h));
    end 
end

% true image
% m = (x > .2).*(x < .3) + sin(4*pi*x).*(x > 0.5) + 0.0 * cos(30*pi*x);
m = 0.75.*( 0.1 < x & x < 0.25) + 0.25.*(0.3 < x & x < 0.32) + ...
    ((sin(2*pi*x)).^4).* (0.5 < x & x < 1) + 0.0 .* (x < 0.1)+ 0.0 .* (0.25 <= x & x <= 0.3)...
    + 0.0 .* (0.32 <= x & x <= 0.5);
% convolved image
d = K * m;
noise_level = sqrt(0.1) * randn(N,1);
% noisy data, noise has sigma (standard deviation) = 0.1
dn = d + noise_level;
plot(x,d,x,dn,'Linewidth', 2);
legend('data', 'noisy data');

%% TSVD
alpha = 1; % 0.0001,0.001,0.1,1;
[U,S,V] = svd(K);
S = S.*(S>sqrt(alpha));
m_alpha = V*pinv(S)*U'*dn ;
plot(x,m,x,m_alpha,'Linewidth', 2);
legend('m_true', 'm_alpha');
title("? = 1");

%% Tikhonov regularization parameter
alpha = 1; % 0.0001,0.001,0.1,1

% solve Tikhonov system
m_alpha = (K'*K + alpha * eye(N))\(K'*dn);
% comment out next 3 if you dont want figure
figure;
plot(x,m,x,m_alpha,'Linewidth', 2), axis([0,1,-1.5,1.5]);
legend('exact data', 'Tikhonov reconstruction');
title("? = 1");

%% plot L-curve
alpha_list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1, 1, 1e1, 1e2, 1e3, 1e4];
no = length(alpha_list);
misfit = zeros(no,1);
reg = zeros(no,1);

for k = 1:no
    alpha = alpha_list(k);
    m_alpha = (K'*K + alpha * eye(N))\(K'*dn);
    misfit(k) = norm(K*m_alpha - dn);
    reg(k) = norm(m_alpha);
end

figure;
loglog(misfit, reg, 'Linewidth', 2);
hold on;
loglog(misfit(6), reg(6), 'ro', 'Linewidth', 3); % optimal alpha = 5e-2;
xlabel('||K*m - d||'); ylabel('||m||');
title("L-curve Method: ? = 0.05");

%% Morozov's discrepancy criterion
alpha_list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1, 1, 1e1, 1e2, 1e3, 1e4];
no = length(alpha_list);
misfit = zeros(no,1);
reg = zeros(no,1);
noise = norm(noise_level)* ones(no,1);

for k = 1:no
    alpha = alpha_list(k);
    m_alpha = (K'*K + alpha * eye(N))\(K'*dn);
    misfit(k) = norm(K*m_alpha - dn);
    reg(k) = norm(m_alpha);
end

figure;
loglog(alpha_list, misfit, 'Linewidth', 2);
hold on;
loglog(alpha_list, noise, 'r--', 'Linewidth', 2);
xlabel('alpha'); ylabel('||K*m - d||');
plot(alpha_list(7), noise(7), 'ro', 'Linewidth', 3);
title("Morozov's discrepancy criterion: ? = 0.1");

%% Generalized cross validation
[U,S,V] = svd(K);
alpha_list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1, 1, 1e1, 1e2, 1e3, 1e4];
no = length(alpha_list);
nu = zeros(no,1);
do = zeros(no,1);
N = 200;
for j = 1:no
    alpha = alpha_list(j);
    m_alpha = (K'*K + alpha * eye(N))\(K'*dn);
    K_alpha = (K'*K + alpha * eye(N))\K';
    misfit = K*m_alpha - dn;
    nu(j) = N* misfit'*misfit;
    do(j) = (N-trace(K*K_alpha))^2;  
end

GCV = nu./do;

figure;
loglog(alpha_list, GCV, 'Linewidth', 2);
hold on;
xlabel('alpha'); ylabel('GCV');
plot(alpha_list(5), GCV(5), 'ro', 'Linewidth', 3);
title("GCV: ? = 0.01");

%% ||m_alpha - m_true||
alpha_list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1, 1, 1e1, 1e2, 1e3, 1e4];
no = length(alpha_list);
misfit = zeros(no,1);

% solve Tikhonov system
for i = 1:no
    m_alpha = (K'*K + alpha_list(i) * eye(N))\(K'*dn);
    misfit(i) = norm(m-m_alpha);
end

figure;
loglog(alpha_list, misfit, 'Linewidth', 2);
hold on;
xlabel('alpha'); ylabel('||m_true - m_?||');
plot(alpha_list(6), misfit(6), 'ro', 'Linewidth', 3);
title("? = 0.05");
