close all
clear all

% Import image
I = imread('circle.png');
m_true = 146*double(I);

% Get the number of pixels in the vertical and horizontal direction (N1, and N2)
[N1, N2] = size(m_true);
N = N1 * N2;

% Generate x and y axis
x=linspace(0,N2,N2);
y=linspace(0,N1,N1);
[xx,yy] = meshgrid(x,y);

% draw the imported image
figure;
colormap gray;
imagesc(m_true);
title('True image');


% Use different Gaussian blurring in x and y-direction
gamma1 = 5;
C1 = 1 / (sqrt(2*pi)*gamma1);
gamma2 = 12;
C2 = 1 / (sqrt(2*pi)*gamma2);

% blurring operators for x and y directions
K1 = zeros(N1,N1);
K2 = zeros(N2,N2);
for l = 1:N1
    for k = 1:N1
    	K1(l,k) = C1 * exp(-(l-k)^2 / (2 * gamma1^2));
    end
end
for l = 1:N2
    for k = 1:N2
    	K2(l,k) = C2 * exp(-(l-k)^2 / (2 * gamma2^2));
    end
end

% blur the image: first, K2 is applied to each column of I,
% then K1 is applied to each row of the resulting image
Ib = (K2 * (K1 * m_true)')';

% plot blurred image
figure;
colormap gray;
imagesc(Ib);
title('blurred image');

% add noise and plot noisy blurred image
Ibn = Ib + 16 * randn(N1,N2);
figure;
colormap gray;
imagesc(Ibn);
title('blurred noisy image');

% compute Tikhonov reconstruction with regularization
% parameter alpha, i.e. compute m = (K'*K + alpha*I)\(K'*d)

% first construct the right hand side K'*d
K_Ibn = (K2 * (K1 * Ibn)')';

% then set the regularization parameter 
alpha = 1.5e-3;

% now solve the regularized inverse problem to reconstruct the 
% the image using preconditioned conjugate gradients (pcg) to solve the
% system in a matrix-free way using function "apply"

m_alpha = pcg(@(in)apply(in,K1,K2,N1,N2,alpha),K_Ibn(:),1e-6,1500);
figure;
%colormap gray;
imagesc(reshape(m_alpha,N1,N2));
title('Tikhonov reconstruction');

%% plot L-curve
alpha_list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1, 1, 1e1, 1e2, 1e3, 1e4];
no = length(alpha_list);
misfit = zeros(no,1);
reg = zeros(no,1);

for k = 1:no
    alpha = alpha_list(k);
    m_alpha = pcg(@(in)apply(in,K1,K2,N1,N2,alpha),K_Ibn(:),1e-6,1500);
    m_alpha = reshape(m_alpha,N1,N2);
    I_alpha = (K2 * (K1 * m_alpha)')';
    misfit(k) = norm(I_alpha - Ibn);
    reg(k) = norm(m_alpha);
end

figure;
loglog(misfit, reg, 'Linewidth', 2);
hold on;
loglog(misfit(5), reg(5), 'ro', 'Linewidth', 3);
xlabel('||K*m - d||'); ylabel('||m||');
title("L-curve")

m_alpha_recon = pcg(@(in)apply(in,K1,K2,N1,N2,reg(5)),K_Ibn(:),1e-6,1500);
figure;
%colormap gray;
imagesc(reshape(m_alpha_recon,N1,N2));
title('Tikhonov reconstruction: ? = 0.01');
%% ||m_alpha - m_true||
alpha_list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1, 1, 1e1, 1e2, 1e3, 1e4];
no = length(alpha_list);
misfit = zeros(no,1);

% solve Tikhonov system
for i = 1:no
    alpha = alpha_list(i);
    m_alpha = pcg(@(in)apply(in,K1,K2,N1,N2,alpha),K_Ibn(:),1e-6,1500);
    m_alpha = reshape(m_alpha,256,256);
    misfit(i) = norm(m_true-m_alpha);
end

figure;
loglog(alpha_list, misfit, 'Linewidth', 2);
hold on;
xlabel('alpha'); ylabel('||m_true - m_?||');
plot(alpha_list(5), misfit(5), 'ro', 'Linewidth', 3);
title("? = 0.01");