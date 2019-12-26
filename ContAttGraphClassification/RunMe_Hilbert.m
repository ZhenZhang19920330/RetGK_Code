clc; close all; clear;

% Synthie           Nstep=50, finalkernel="Gaussian"  97.1(0.28)
% ENZYMES           Nstep=20, finalkernel="Gaussian"  70.03(0.91)
% FRANKENSTEIN      Nstep=50, finalkernel="Laplacian" 76.35(0.26)
% SYNTHETICnew      Nstep=50, finalkernel="Gaussian"  97.93(0.34)
% PROTEINS          Nstep=50, finalkernel="Laplacian" 76.51(0.60)
%% Parameters:
datasetName ='PROTEINS';Nstep=50; finalkernel="Laplacian";



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
clear s;
sigmaAtt=sqrt(DimAtt);
if strcmp(datasetName,'FRANKENSTEIN')
    sigmaAtt=sqrt(1/0.0073);
end
%% N_step feature extraction

FeatureSets=cell(NumGraph,1);
NodeAttTotal=cell(NumGraph,1);
NodeLabTotal=cell(NumGraph,1);
y=zeros(NumGraph,1);
for i=1:NumGraph
    [FeatureSets{i},NodeLabTotal{i},NodeAttTotal{i}]=NstepFeature(Nstep,Graphs(i),NodeLab_Exist,NodeAtt_Exist);
    y(i)=Graphs(i).label;
end
clear Graphs;
%% Compute the first kernels width for RPF
tic;
NumSampleEach=5;
SampleTotal=[];
for i=1:NumGraph   
    [~,N]=size(FeatureSets{i});
    if N>=NumSampleEach
        Ind=randsample(N,NumSampleEach);
        SampleTotal=[SampleTotal FeatureSets{i}(:,Ind)];
    end   
end
DistSampleTotal=pdist2(SampleTotal',SampleTotal');
sigmaFeat=median(DistSampleTotal(:));  % width for the FIRST kernel


%% Compute the pairwise MMD distance between graphs
SelfTerm=zeros(NumGraph, 1);
for i=1:NumGraph
    XFeat=FeatureSets{i}; [~,Nx]=size(XFeat); XAtt=NodeAttTotal{i};XLab=NodeLabTotal{i};
    DistxxFeat=pdist2(XFeat',XFeat');
    DistxxAtt=pdist2(XAtt',XAtt');
    KxxFeat=exp(-(DistxxFeat/sigmaFeat));
    KxxAtt=exp(-(DistxxAtt/sigmaAtt));
    Kxx=KxxFeat.*KxxAtt;    
    SelfTerm(i)=sum(Kxx(:))/(Nx*Nx);
end


MMDist=zeros(NumGraph);
for i=1:NumGraph
    
    for j=i+1:NumGraph
        XFeat=FeatureSets{i}; [~,Nx]=size(XFeat); XAtt=NodeAttTotal{i};
        YFeat=FeatureSets{j}; [~,Ny]=size(YFeat); YAtt=NodeAttTotal{j};
        DistxyFeat=pdist2(XFeat',YFeat');
        DistxyAtt=pdist2(XAtt',YAtt');
        KxyFeat=exp(-(DistxyFeat/sigmaFeat));
        KxyAtt=exp(-(DistxyAtt/sigmaAtt));
        Kxy=KxyFeat.*KxyAtt;
        MMDist(i,j)=SelfTerm(i)+SelfTerm(j)-2*sum(Kxy(:))/(Nx*Ny);
    end
    
end
MMDist=MMDist+MMDist';
%% Compute the pairwise kernel
sigma2=median(MMDist(:));
if strcmp(finalkernel,"Gaussian")
    K=exp(-MMDist/sigma2);
end
if strcmp(finalkernel,"Laplacian")
    K=exp(-sqrt(MMDist/sigma2));
end
toc;
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
