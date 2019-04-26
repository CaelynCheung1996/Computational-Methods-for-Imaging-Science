% % % % % % % % % % % % % % 
% Xiaohui Zhang
% Assignment 3
% Mar 22, 2019
% % % % % % % % % % % % % % 
%% Problem 2
clear all;
clc;

syms x y;
gamma = 100;
f(x,y) =gamma*(x^2-y)^2+(x-1)^2;

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
tol = 10e-8;

%% SGD with exact line search
x0 = 1; y0 = 1.1; % initial guess
Xk(1) = x0;
Xk(2) = y0;
gradient_plot1 = [];
sequence1 = [];
for k = 0:max_iter
    Gk(1) = gx(Xk(1),Xk(2));
    Gk(2) = gy(Xk(1),Xk(2));
    gradient_plot1 = [gradient_plot1,norm(Gk)];
    sequence1 = [sequence1, Xk];
    if norm(Gk) <= tol
        break;
    else
        pk = -Gk;
        evalfun = @(alpha)((Xk(1)+alpha*pk(1))^2-(Xk(2)+alpha*pk(2)))^2 + ((Xk(1)+alpha*pk(1))-1)^2;
        alpha_k = fminbnd(evalfun,-3,3);
        Xk = Xk + alpha_k * pk;
        disp(alpha_k);
    end
end

h = plot(gradient_plot1, 'LineWidth', 2);
xlabel('iteration');ylabel('gradient');
title('SGD with exact line search');

% plot Rosenbrock's function
subplot(1,2,1);
x = linspace(-1.5,2);
y = linspace(-0.5,3);
[X,Y] = meshgrid(x,y);
Z = gamma*(X.^2-Y).^2+(X-1).^2;
contour(X,Y,log(Z),'LineWidth',2);
%contour(X,Y,log(Z),20);
hold on;
plot(sequence1(1,:), sequence1(2,:),'r','LineWidth',2);
title('steepest descent, gamma = 100');
hold on;
%% NM with exact line search
x0 = 0; y0 = 0; % initial guess H is pd
Xk(1) = x0;
Xk(2) = y0;
gradient_plot2 = [];
sequence2 = [];

for k = 0:max_iter
    % gradient
    Gk(1) = gx(Xk(1),Xk(2));
    Gk(2) = gy(Xk(1),Xk(2));
    % hessian
    Hk(1,1) = hx(Xk(1),Xk(2));
    Hk(2,2) = hy(Xk(1),Xk(2));
    Hk(1,2) = hxy(Xk(1),Xk(2));
    Hk(2,1) = hyx(Xk(1),Xk(2));
    gradient_plot2 = [gradient_plot2,norm(Gk)];
    sequence2 = [sequence2, Xk];

    if norm(Gk) <= tol
        break;
    else
        pk = -pinv(Hk)*Gk;
        % exact line search
        evalfun = @(alpha)((Xk(1)+alpha*pk(1))^2-(Xk(2)+alpha*pk(2)))^2 + ((Xk(1)+alpha*pk(1))-1)^2;
        alpha_k = fminbnd(evalfun,-2,2);
        Xk = Xk + alpha_k * pk;
    end
end

h = plot(gradient_plot2, 'LineWidth', 2);
xlabel('iteration');ylabel('gradient');
title('Newton method with exact line search');
%saveas(h,'Q2_NM_exactline.png');

% plot Rosenbrock's function
subplot(1,2,2);
x = linspace(-1.5,2);
y = linspace(-0.5,3);
[X,Y] = meshgrid(x,y);
Z = gamma*(X.^2-Y).^2+(X-1).^2;
contour(X,Y,log(Z),'LineWidth', 2);
hold on;
plot(sequence2(1,:), sequence2(2,:),'r', 'LineWidth', 2);
title('Newton method, gamma=100');
hold on;
