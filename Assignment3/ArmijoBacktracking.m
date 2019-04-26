function alpha_k = ArmijoBacktracking(f,xk,gk,pk,alpha_k)
j = 0;
c = 10e-4; 
max_backtracking_iter = 100;
while j < max_backtracking_iter
    if f(Xk(1) + alpha_k, Xk(2) + alpha_k) <= f(Xk(1), Xk(2)) + c*alpha_k*Gk'*pk
        return
    else
        alpha_k = alpha_k/2;
        j = j + 1;
    end
end
end
