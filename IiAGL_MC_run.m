% %Inclusivity-induced Adaptive Graph Learning for Multi-view Clustering 
% %The code is written by Xin Zou on 2022/05/15.
%%
clearvars;clc;
addpath('./Evaluation/');
dataPath = './Datasets/'; 
addpath(dataPath);
% 
%% Read data  
%

% Graph paras
options = [];
options.NeighborMode = 'KNN';
options.WeightMode = 'Binary';
options.k = 7;
SaveResults = false;
% % effecitve
% dataName = 'LandUse-21'; mu = 0.01;beta = 1; % maxIter = 4; 
% dataName = '100leaves'; mu = 0.001; beta = 1; %options.k = 40;
% dataName = 'MSRCv1'; mu = 0.1; options.k = 16;% mu = 0.1; options.k = 16;
% dataName = 'BBCSport'; mu = 0.01;  beta = 10;
% dataName = 'CiteSeer'; mu = 0.01; 
% dataName = 'Cora'; mu = 0.01;  
% dataName = 'BBC'; mu = 1e-3; beta = 10; 
% dataName = '3sources'; mu = 1;  eta = 1;
% dataName = 'scene-15'; mu = 0.01; options.k = 18; % options.k = 25;
% dataName = 'WikipediaArticles'; mu = 0.009; beta = 1;
% % normal
% dataName = 'HW2sources'; mu = 0.01;  %options.k = 40;
% dataName = 'ORL'; mu = 0.0001; 
% dataName = 'Caltech101-20'; mu = 0.01; 
% dataName = 'WebKB'; mu = 0.1; options.k = 22;
% dataName = 'Caltech101-7'; mu = 0.001;
% dataName = 'Handwritten'; mu = 1;maxIter = 10;beta = 1;



dataName = 'Handwritten';
mu =0.001;  beta = 1;
maxIter = 1;
%
% mu = [1e3 1e2 1e1 1e0 1e-1 1e-2 1e-3];
% SaveResults = true;
% beta = [1e3 1e2 1e1 1e0 1e-1 1e-2 1e-3]; 
%
dataset = load([dataPath dataName '.mat']);
% X = dataset.fea;      %sample
% gt = dataset.gt;     %groudtruth
X = dataset.X;      %sample
gt = dataset.Y;     %groudtruth
if min(gt)==0
    gt=gt+1;
end
%% Setting parameter

c = length(unique(gt)); %number of clusters
v = length(X);          %number of views
n = length(gt);         %number of samples
fprintf('====== Current dataset:  %s ( %d samples, %d class, %d views ) =====\n',dataName, n, c, v);




%% Initialization
tic;
NorX = cell(1, v);
S = cell(size(X));
for i = 1 : v
%     X{i} = mapstd(X{i},0,1);    % X{i} is m * N
    tempData = NormalizeFea(X{i});                 
    NorX{i} = tempData;
    tempData = ConstructW(tempData, options);
    S{i} = full(tempData);
end
X = NorX;
clear NorX tempData;
[~, num] = size(X{1});
for i = 1 : v
    X{i} = X{i}';
end
clear num;

%% Iteration
res = [];
for i = 1 : length(mu)
    for j = 1 : length(beta)
        fprintf('====== mu: %f , beta: %f =====\n', mu(i), beta(j));
        [U, label, F, w, alpha, Z, E, best, object, y_acc] = IiAGL_MC_main(X, S, v, c, n, mu(i), beta(j), maxIter, gt);
        res = [res;mu(i) beta(j) best(7) best(4) best(5) best(1) best(2) best(3) best(8)];
    end
end
if(SaveResults)
    save([dataName '12_result.mat'], 'res');
end
toc;
for i = 1 : v
    figure(i);
    imagesc(S{i});
end
% for i = 1 : v
%     figure(v+i);
%     imagesc(Z{i});
% end    

% for i = 1 : v
%     figure;
%     imagesc(E{i});
% end
figure(2*v+1);
imagesc(U);
fprintf('ACC: %f \n',best(7));
fprintf('NMI: %f \n',best(4));
fprintf('ARI: %f \n',best(5));
fprintf('Purity: %f \n',best(8));
fprintf('Precision: %f \n',best(2));
fprintf('Recall: %f \n',best(3));
%load('cm2.mat');
%colormap(cm2);