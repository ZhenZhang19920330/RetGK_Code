clc; close all; clear;


% ENZYMES       Nstep=50, D=200, gamma=10, 59.05(1.14)
% MUTAG         Nstep=50, D=200, gamma=10, 90.06(0.96)
% PROTEINS      Nstep=50, D=200, gamma=10, 75.23(0.34)
% DD            Nstep=50, D=200, gamma=10, 81.00(0.51)
% NCI1          Nstep=50, D=200, gamma=10, 83.51(0.20)
% PTC_FM        Nstep=50, D=200, gamma=100,63.23(1.26)
% PTC_FR        Nstep=50, D=200, gamma=100,67.83(1.06)
% PTC_MM        Nstep=50, D=200, gamma=100,67.85(1.40)
% PTC_MR        Nstep=50, D=200, gamma=10, 61.12(1.54)

%% Parameters:
datasetName='PTC_MR'; Nstep=50;D=200;gamma=100; 
s = load([datasetName '.mat']); 
% The dataset has the following structure:
% 1. s.n_graphs: the number of graphs;  
% 2. s.n_classes: the number of classes; 
% 3. s.has_node_labels: denotation of the existence of node labels (Discrete)
% 4. s.has_node_attributes: denotation of the existence of node attributes
%    (Continuous)
% 5. s.unique_node_labels: the set of nodel labels 
% 6. s.n_attributes_per_node:  the dimension of attributes 
% 7. s.graphs: the structure of graphs
%    a>  s.graphs(i).n_node: the node number of graph i
%    b>  s.graphs(i).n_edges: the edge number of graph i
%    c>  s.graphs(i).A: the adjacent matrix of graph i
%    d>  s.graphs(i).label: the class of graph i
%    e>  s.graphs(i).node_labels: an s.graphs(i).n_node*1 vector storing the
%        node label information of graph i
%    f>  s.graphs(i).node_attributes: an
%        s.graphs(i).n_node*s.n_attributes_per_node matrix storing the node
%        attribute information of graph i


%%
NumGraph=s.([datasetName '_data']).n_graphs;
NodeLab_Exist=s.([datasetName '_data']).has_node_labels;
NodeAtt_Exist=s.([datasetName '_data']).has_node_attributes;
DimAtt=s.([datasetName '_data']).n_attributes_per_node;
Graphs=s.([datasetName '_data']).graphs;
MaxLabel=max(s.([datasetName '_data']).unique_node_labels)+1;
clear s;
%% N_step feature extraction
tic;
FeatureSets=cell(NumGraph,1);
NodeAttTotal=cell(NumGraph,1);
NodeLabTotal=cell(NumGraph,1);
y=zeros(NumGraph,1);
for i=1:NumGraph
    [FeatureSets{i},NodeLabTotal{i},NodeAttTotal{i}]=NstepFeature(Nstep,Graphs(i),NodeLab_Exist,NodeAtt_Exist); 
    y(i)=Graphs(i).label;
end
save('ENZYMESlabels','y');
clear Graphs;
%% approximately represent each graph by an D*MaxLabel Cross-correlation operator (matrix)

W=gamma*randn(Nstep,D);
B=rand(D,1)*2*pi;
VecCCO=zeros(D*(MaxLabel+1),NumGraph);
for i=1:NumGraph
    X=FeatureSets{i};NodeLab=NodeLabTotal{i};
    [~,N]=size(X);
    Z=sqrt(2/D)*cos(W'*X+B*ones(N,1)'); 
    % each column of Z is the representation of ith node's Nstep-feature.
    % generate matrix E, the ith row of E denotes the (discrete) label of
    % node i
    E=zeros(N,MaxLabel+1);
    for j=1:N
        E(j,NodeLab(j)+1)=1;
    end
    VecCCO(:,i)=(reshape(Z*E,[],1))/N;
end
VecCCO=sparse(VecCCO);

clear FeatureSets;clear NodeAttTotal; clear NodeLabTotal;


%% Compute the kernel matrix

DistMat=pdist2(VecCCO',VecCCO');% Laplacian Kernel
clear VecCCO;
sigma=median(DistMat(:));
K=exp(-(DistMat/sigma));
%K=normalize_kernel(K);% Normalized kernel
toc;
clear DistMat;
%% 10-folds cross-validation SVM
NumExp=10;
accuracy = zeros(NumExp,1);
for i=1:NumExp
    accuracy(i)=TenFoldCvSvm(K,y);
end
MeanAcc=mean(accuracy);StdErr=std(accuracy);
disp('========================================================');
fprintf('The mean accuracy is %f, and the standard error is %f\n',MeanAcc, StdErr);
disp('========================================================');
