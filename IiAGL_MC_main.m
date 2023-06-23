% %Inclusivity-induced Adaptive Graph Learning for Multi-view Clustering
% %The code is written by Xin Zou on 2022/05/15.
%%
function [U, label, F, w, alpha, Z, E, best, object, y_acc, y_nmi] = IiAGL_MC_main(X, Z_init, v, c, n, mu, beta, maxIter, gt)
rho = 1.1;
delta = 1e-3;
omiga = 1e-4;
maxIter = 1;
% % initialize Z{i}, P{i}, w, alpha
Z = Z_init;
P = Z;
U = Z{1};
w = ones(v, 1)/v;
alpha = ones(v, 1);
for i = 1 : v
    E{i} = zeros(size(X{i}));
    N{i} = zeros(size(Z{i}));
    M{i} = zeros(size(X{i}));
end
best = zeros(1,8);
eta = 1;
object = zeros( maxIter, 1 );
y_acc = zeros(maxIter, 1);
y_nmi = zeros(maxIter, 1);
% % iterate
for iter = 1 : maxIter
    fprintf('----------\n ');
    disp(['iter:', num2str(iter)]);
    knn_idx = true(n);
    b = w(1);
    for j = 2 : v
       b = b + w(j);
    end
    U = zeros(size(U));
    for j = 1 : v
       U = U + w(j) * alpha(j) * Z{j};
    end
    U = U / b;
    % %
    % %
    %Update Z{i}, fix the other variables;
    alpMat = alpha * alpha';
    wMat = w * w';
    for i = 1 : length(w)
        alpMat(i, i) = 0;
        wMat(i, i) = 0;
    end
    I = eye(size(wMat));
    for i = 1 : v
       tmp = beta*w(i)*alpha(i)*U +delta*(X{i}'*(X{i} - E{i} + M{i}/delta)+P{i}-N{i}/delta);
       R{i} = tmp(knn_idx);
       tmp = X{i}'*X{i}*Z{i};
       XXZ{i} = delta*tmp(knn_idx);
    end
    Q = beta*diag(w).*alpMat - mu* alpMat .* wMat + delta*I;
    coR = cat(2, R{:})';
    coXXZ = cat(2, XXZ{:})';
    if det(Q) == 0
       solution = (pinv(Q) * (coR - coXXZ));
       fprintf('\n ------------- \ns');
    else
       solution = abs((Q \ (coR - coXXZ)));
    end
    solution(solution < 0) = 0;
    for i = 1 : v
       tmp = solution(i,:);
       Z{i} = zeros(n, n);
       Z{i}(knn_idx) = tmp;
    end
    clear solution tmp Q coR coXXZ;
    % %
    % 
    % Update P{i}, fix the other variables;
    for i = 1 : v
       Z{i}(isnan(Z{i})) = 0;
       temp = Z{i} + N{i}./delta;
       [pU, Sgm, pV] = svd(temp, 'econ');
       pU(isnan(pU)) = 0;
       Sgm(isnan(Sgm)) = 0;
       pV(isnan(pV)) = 0;
       Sgm = diag(Sgm);
       svp = length(find(Sgm > 1/delta));
       if svp>=1
           Sgm = Sgm(1:svp) - 1/delta;
       else
           svp = 1;
           Sgm = 0;
       end
       P{i} = pU(:,1:svp) * diag(Sgm) * pV(:,1:svp)';
    end
    clear temp pU Sgm pV svp;
    % %
    % 
    % Update E{i}, fix the other variables;
    for i = 1 : v
       Xv=X{i};
       tempE=Xv-Xv*Z{i}+M{i}./delta;
       tmp = eta*w(i)/delta;
       E{i} = max(0, tempE - tmp) + min(0, tempE + tmp);
    %        E{i}=Solve_L21F(tempE, eta*w(i)/delta);
    end
    clear Xv tempE;
    % %
    % %
    % Update w, fix the other variables;
    D = zeros(v);
    for i = 1 : v
       for j = i : v
           D(i, j) = sum(sum(Z{i}.*Z{j}));
           D(j, i) = D(i, j);
       end
    end
    aad = alpMat .* D ;
    for i = 1 : v
       E_l21 = 0.0;
       for j = 1 : n
           E_l21 = E_l21 + norm(E{i}(:,j),2);
       end
       g{i} = E_l21 +(0.5) * beta * norm((U - alpha(i) * Z{i}), 'fro');
       XXZ{i} = X{i}'*X{i}*Z{i};
    end
    coG = cat(2, g{:})';
    if det(aad) == 0
       solution = (pinv(aad) * (coG));
       fprintf('\n ------------- \n');
    else
       solution = abs((aad \ (coG)));
    end
    solution(solution < 0) = 0;
    for i = 1 : v
       w(i) = solution(i);
    end
    clear solution aad coG;
    % %
    % 
    % Update alpha, fix the other variables;
    I = eye(size(wMat));
    J = (beta*diag(w) - ((mu*b+beta)/b)*wMat);
    one = ones(v, 1);
    tmp = [J.*D, one; one', 0];
    rb = [zeros(v ,1); 1];
    if det(tmp) == 0
       solution = pinv(tmp) * rb;
    else
       solution = abs(tmp \ rb);
    end   
    solution(solution < 0) = 0;
%     alpha = EProjSimplex_new(solution(1:v));
    alpha = solution(1:v);
    clear solution;
    % Update N and M;
    for i = 1 : v
       leq1 = Z{i} - P{i};
       leq2 = X{i} - X{i}*Z{i} - E{i};
       N{i} = N{i} + delta * leq1;
       M{i} = M{i} + delta * leq2;
    end
    % Update Y;
    %Update U, fix the other variables;
    tmp = zeros(size(U));
    for j = 1 : v
       tmp = tmp + w(j) * alpha(j) * Z{j};
    end
    U = tmp / b;

    % Update delta
    delta = min(rho * delta, 1e6);
    
    % Spectral Clustering
    [label, F] = SpectralClustering(U, c, 3);
    result = Clustering8Measure(gt, label);
    
    y_acc(iter) = result(7) ;
    y_nmi(iter) = result(4) ;
    % update object
    term1 = 0;
    term2 = 0;
    term3 = 0;
    for i = 1 : v
        [~,s,~] = svd(Z{i},'econ');
        term1 = term1 + sum(diag(s));
        tmp = 0;
        for j = 1 : v
            tmp = tmp + norm(E{i}(:, j), 2);
        end
        term1 = term1 + w(i) * tmp;
        for j = 1 : v
            term2 = term2 + w(i) * w(j) * norm((alpha(i)*Z{i}).*(alpha(j)*Z{j}), 1);
        end
        term3 = term3 + w(i) * norm(U - alpha(i) * Z{i}, 'fro')^2;
    end
    object(iter) = term1 + 0.5 * mu * term2 + 0.5 * beta * term3;
    if iter > 1 && abcr >= result(7) && (result(7)-abcr) < omiga*abcr
        U = u1;label = l1; w=w1; Z=z1;
        break;
    end
    abcr = result(7);
    best = result;
    u1 = U;l1 = label;w1 = w;z1 = Z;
    % convergence
%     if iter > 40 && abs((obj(iter) - obj(iter - 1)) / obj(iter - 1)) < 1e-6
%         break;
%     end
    fprintf('ACC: %f \n',result(7));
    fprintf('NMI: %f \n',result(4));
    fprintf('AR: %f \n',result(5));
    fprintf('Fscore: %f \n',result(1));
    fprintf('Precision: %f \n',result(2));
    fprintf('Recall: %f \n',result(3));
    fprintf('Purity: %f \n',result(8));
end
%
%
end
