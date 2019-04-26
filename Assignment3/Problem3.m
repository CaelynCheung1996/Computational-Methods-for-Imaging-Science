% % % % % % % % % % % % % % 
% Xiaohui Zhang
% Assignment 3
% Mar 22, 2019
% % % % % % % % % % % % % % 
%% Problem 3
clear all;
clc;

% Inexact Newton CG method
newton_maxiter = 500;
tol = 10e-8;
n = 1000;
x0 = zeros(n,1);   
p0 = zeros(n,1);
gradient_plot6 = [];
[f0, g0, H0] = objectiveFunction(x0);
xk = x0;

for k = 0:newton_maxiter
    % compute gradient
    [fk, gk, Hk] = objectiveFunction(xk);

    if norm(gk)/norm(g0) <= tol
        break;
    end
    
    % select tolerance ita_k
    %ita_k = 0.5;
    %ita_k = min(0.5,sqrt(norm(gk)/norm(g0)));
    ita_k = min(0.5,norm(gk)/norm(g0));
    
    % use CG to find a descent direction
    gc_iter = 1000; % num of gc iteration
    [pk, i] = cg_steihaug(Hk,-gk,gc_iter,ita_k,zeros(n,1));
    gradient_plot6 = [gradient_plot6,norm(gk)];

    % Armijo backtracking
    j = 0;
    c = 10e-4;
    max_backtracking_iter = 100;
    alpha0 = 1; % initial alpha
    alpha_k = alpha0;
    while j < max_backtracking_iter
        [fk, gk, Hk] = objectiveFunction(xk);
        [f_alphak, g_alphak, H_alphak] = objectiveFunction(xk+alpha_k*pk);
        if f_alphak <= fk + c*alpha_k*gk'*pk
            break;
        else
            alpha_k = alpha_k/2;
            j = j + 1;
        end
    end
    
    xk = xk + alpha_k * pk;
end

semilogy(gradient_plot4, 'LineWidth', 2);
hold on;
semilogy(gradient_plot5, 'LineWidth', 2);
hold on;
semilogy(gradient_plot6, 'LineWidth', 2);
%legend('n=1000, ?=0.5','n=1000, ?=min(0.5,sqrt(||gk||/||g0||))','n=1000, ?=min(0.5,||gk||/||g0||)');
legend('n=250, ?=min(0.5,||gk||/||g0||)','n=500, ?=min(0.5,||gk||/||g0||)','n=1000, ?=min(0.5,||gk||/||g0||)');
xlabel('iteration');ylabel('||gk||');
hold on;
