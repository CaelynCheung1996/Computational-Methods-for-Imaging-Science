% %%%%%%%%%%%%%%%%%%%%%%%
% Xiaohui Zhang
% Assignment 3
% Mar 21, 2019
% %%%%%%%%%%%%%%%%%%%%%%%
%% Problem 1
clear all;
clc;
%% 1A 
x = linspace(-pi/2,pi/2);
y = linspace(-10*pi/2,10*pi/2);
[X,Y] = meshgrid(x,y);
Z = cos(X-Y/10).*cos(X+Y/10);
imagesc(x, y, Z);
n = size(x,2);
region = zeros(n,n);
for i = 1:n
    for j = 1:n
        if Z(i,j) > 0
            region(i,j) = 1;
        end
    end
end
imagesc(x, y, region);


%% 1D
syms x y;
f(x,y) = -cos(x)*cos(y/10);

% gradient & hessian
gx = diff(f,x);
gy = diff(f,y);
hx = diff(gx,x);
hy = diff(gy,y);
hxy = diff(gx,y); 
hyx = diff(gy,x);

Xk = zeros(2,1);
Gk = zeros(2,1);
Hk = zeros(2,2);

max_iter = 500;
tol = 1e-6;

%% SGD without line search
alpha = 0.1; % fixed step size
x0 = 1; y0 = 1; % initial guess H is pd
Xk(1) = x0;
Xk(2) = y0;
gradient_plot1 = [];
for k = 1:5000
    Gk(1) = gx(Xk(1),Xk(2));
    Gk(2) = gy(Xk(1),Xk(2));
    gradient_plot1 = [gradient_plot1,norm(Gk)];
    if norm(Gk) <= tol
        return
    else
        pk = -Gk;
        Xk = Xk + alpha * pk;
    end
end
h = plot(gradient_plot1, 'LineWidth', 2);
xlabel('iteration');ylabel('gradient');
title('SGD without line search');
saveas(h,'SGD_noline.png');

%% SGD with exact line search
x0 = 0; y0 = 0;%initial guess
Xk(1) = x0;
Xk(2) = y0;
gradient_plot2 = [];

for k = 0:max_iter
    Gk(1) = gx(Xk(1),Xk(2));
    Gk(2) = gy(Xk(1),Xk(2));
    gradient_plot2 = [gradient_plot2,norm(Gk)];
    if norm(Gk) <= tol
        return
    else
        pk = -Gk;
        fun = @(alpha)(-cos(Xk(1)+alpha*pk(1)))*cos((Xk(2)+alpha*pk(2))/10);
        alpha_k = fminbnd(fun,-3,3);
        Xk = Xk + alpha_k * pk;
    end
end

h = plot(gradient_plot2 ,'LineWidth', 2);
xlabel('iteration');ylabel('gradient');
title('SGD with exact line search');

%% NM without line search
alpha = 0.1; % fixed step size
x0 = 1; y0 = 1; % initial guess
Xk(1) = x0;
Xk(2) = y0;
gradient_plot3 = [];

for k = 0:max_iter
    % gradient
    Gk(1) = gx(Xk(1),Xk(2));
    Gk(2) = gy(Xk(1),Xk(2));
    % hessian
    Hk(1,1) = hx(Xk(1),Xk(2));
    Hk(2,2) = hy(Xk(1),Xk(2));
    Hk(1,2) = hxy(Xk(1),Xk(2));
    Hk(2,1) = hxy(Xk(1),Xk(2));
    gradient_plot3 = [gradient_plot3,norm(Gk)];
    
    if norm(Gk) <= tol
        return
    else
        pk = -inv(Hk)*Gk;
        Xk = Xk + alpha * pk;
    end
end
h = plot(gradient_plot3, 'LineWidth', 2);
xlabel('iteration');ylabel('gradient');
title('Newton method without line search');
saveas(h,'NM_noline.png');

%% NM with exact line search
% x0 = pi; y0 = 2*pi; % initial guess H isn't pd
x0 = 0; y0 = 0; 
Xk(1) = x0;
Xk(2) = y0;
gradient_plot4 = [];

for k = 0:max_iter
    % gradient
    Gk(1) = gx(Xk(1),Xk(2));
    Gk(2) = gy(Xk(1),Xk(2));
    % hessian
    Hk(1,1) = hx(Xk(1),Xk(2));
    Hk(2,2) = hy(Xk(1),Xk(2));
    Hk(1,2) = hxy(Xk(1),Xk(2));
    Hk(2,1) = hyx(Xk(1),Xk(2));
    gradient_plot4 = [gradient_plot4,norm(Gk)];

    if norm(Gk) <= tol
        return
    else
        pk = -inv(Hk)*Gk;
        % exact line search
        fun = @(alpha)(-cos(Xk(1)+alpha*pk(1)))*cos((Xk(2)+alpha*pk(2))/10);
        alpha_k = fminbnd(fun,-2,2);
        Xk = Xk + alpha_k * pk;
    end
end
h = plot(gradient_plot4, 'LineWidth', 2);
xlabel('iteration');ylabel('gradient');
title('Newton method with exact line search');
saveas(h,'NM_exactline_Hpd.png');

