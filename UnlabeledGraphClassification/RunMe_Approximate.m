clc; close all; clear;
% IMDB_BINARY; REDDIT_BINARY; COLLAB; REDDIT_MULTI_5K; REDDIT_MULTI_12K; IMDB_MULTI


% IMDB_BINARY      Nstep=50,D=200,gamma=10;72.27(0.61)
% IMDB_MULTI       Nstep=50,D=200,gamma=10;48.69(0.61)
% COLLAB           Nstep=50,D=200,gamma=10;80.57(0.26)
% REDDIT_BINARY    Nstep=50,D=200,gamma=10;91.61(0.24)
% REDDIT_MULTI_5K  Nstep=50,D=200,gamma=10;55.27(0.34)
% REDDIT_MULTI_12K Nstep=50,D=200,gamma=10;47.14(0.26)

%% Parameters:
datasetName = 'IMDB_BINARY';Nstep=50;D=200;gamma=10; 
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
%    e>  s.graphs(i).node_labels: an (s.graphs(i).n_node*1) vector storing the
%        node label information of graph i
%    f>  s.graphs(i).node_attributes: an
%        (s.graphs(i).n_node*s.n_attributes_per_node) matrix storing the node
%        attribute information of graph i


%%
NumGraph=s.([datasetName '_data']).n_graphs;
NodeLab_Exist=s.([datasetName '_data']).has_node_labels;
NodeAtt_Exist=s.([datasetName '_data']).has_node_attributes;
NumLab=length(s.([datasetName '_data']).unique_node_labels);
DimAtt=s.([datasetName '_data']).n_attributes_per_node;
Graphs=s.([datasetName '_data']).graphs;
clear s;
%% N_step feature extraction
tic;
FeatureSets=cell(NumGraph,1);
NodeAttTotal=cell(NumGraph,1);
NodeLabTotal=cell(NumGraph,1);
y=zeros(NumGraph,1);
for i=1:NumGraph
    [FeatureSets{i},NodeAttTotal{i},NodeLabTotal{i}]=NstepFeature(Nstep,Graphs(i),NodeLab_Exist,NodeAtt_Exist); 
    y(i)=Graphs(i).label;
end

clear Graphs;
%% Random Fourier feature  representation: 
% represent each graph with an element with an D-dimensional vector 

W=gamma*randn(Nstep,D);
B=rand(D,1)*2*pi;
RandFeaturesMMD=zeros(D,NumGraph);
for i=1:NumGraph
    [~,N]=size(FeatureSets{i});
    Z=sqrt(2/D)*cos(W'*FeatureSets{i}+B*ones(N,1)'); 
    % each column of Z is the representation of ith node's Nstep-feature. 
    RandFeaturesMMD(:,i)=Z*ones(N,1)/N;
end

clear FeatureSets;clear NodeAttTotal; clear NodeLabTotal;
%% Compute the kernel matrix
DistMat=pdist2(RandFeaturesMMD',RandFeaturesMMD');%Laplacian Kernel
sigma=median(DistMat(:));
K=exp(-(DistMat/sigma));
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
