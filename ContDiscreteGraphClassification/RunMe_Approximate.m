clc; close all; clear;
% ENZYMES    Nstep=50,DFeat=DAtt=100,gammaFeat=1,  gammaAtt=0.1, 70.63(0.72)
% PROTEINS   Nstep=50,DFeat=DAtt=100,gammaFeat=10, gammaAtt=1,   77.32(0.52)
% BZR        Nstep=50,DFeat=DAtt=100,gammaFeat=10, gammaAtt=1,   87.13(0.72)
% COX2:      Nstep=50,DFeat=DAtt=100,gammaFeat=10, gammaAtt=0.1, 81.43(0.64)
% DHFR       Nstep=50,DFeat=DAtt=100,gammaFeat=10, gammaAtt=0.1, 82.50(0.77)

%% Parameters:
datasetName = 'DHFR'; Nstep=50; DFeat=100;DAtt=100;gammaFeat=10;gammaAtt=0.1;

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
NumLab=length(s.([datasetName '_data']).unique_node_labels);
DimAtt=s.([datasetName '_data']).n_attributes_per_node;
Graphs=s.([datasetName '_data']).graphs;
MaxLabel=max(s.([datasetName '_data']).unique_node_labels);
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

clear Graphs;
%% Represent each graph with an Cross-covariance operator (approximate matrix)
% represent each graph with an element with an D-dimensional vector 

WFeat=gammaFeat*randn(Nstep,DFeat);
BFeat=rand(DFeat,1)*2*pi;
WAtt=gammaAtt*randn(DimAtt,DAtt);
BAtt=rand(DAtt,1)*2*pi;
Tensor=zeros(DFeat*DAtt*(MaxLabel+1),NumGraph);
for i=1:NumGraph
    [~,N]=size(FeatureSets{i});NodeLab=NodeLabTotal{i};
    ZFeat=sqrt(2/DFeat)*cos(WFeat'*FeatureSets{i}+BFeat*ones(N,1)'); 
    % each column of ZFeat is the representation of ith node's Nstep-feature. 
    ZAtt=sqrt(2/DAtt)*cos(WAtt'*NodeAttTotal{i}+BAtt*ones(N,1)'); 
    % each column of ZAtt is the representation of ith node's attribute
    ZLab=zeros(MaxLabel+1,N);
    % each column of ZLab is the one-hop encoding vector of ith nodes's
    % labels
    for j=1:N
        ZLab(NodeLab(j)+1,j)=1;
    end
    tensorgraphi=zeros(DFeat*DAtt*(MaxLabel+1),1);
    for j=1:N
        Temp=kron(ZFeat(:,j),ZAtt(:,j));Temp=kron(Temp,ZLab(:,j));%Temp=sparse(Temp);
        tensorgraphi=tensorgraphi+Temp;
    end
    Tensor(:,i)=(tensorgraphi/N);
end

clear FeatureSets;clear NodeAttTotal; clear NodeLabTotal;
%% Compute the kernel matrix

%DistMat=pdist2(Tensor',Tensor');
Tensor=sparse(Tensor);
DistMat=abs(sqrt(EuclideanDist(Tensor,Tensor)));
sigma=median(DistMat(:));
K=exp(-DistMat/sigma);
toc;
clear Tensor;
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
