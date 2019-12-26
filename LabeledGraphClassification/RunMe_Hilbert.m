clc; close all; clear;

% ENZYMES      Nstep=50;        60.37(0.81)%
% MUTAG        Nstep=50;        90.33(1.09)%
% PROTEINS     Nstep=50;        75.79(0.64)%
% PTC_FM       Nstep=50(5,1);   62.32(0.96)%
% PTC_FR       Nstep=50;        66.71(1.44)%
% PTC_MM       Nstep=50;        65.55(1.10)%
% PTC_MR       Nstep=50(2);     63.53(1.63)%
% DD           Nstep=50,        81.60(0.34)%
% NCI1         Nstep=50,        84.49(0.22)%
% NCI109       Nstep=50,        83.82(0.19)%

%% Parameters:
datasetName='PTC_MM';Nstep=50;


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

clear Graphs;
%% Compute the first kernels width

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
%% Compute the self term
SelfTerm=zeros(NumGraph,1);

for i=1:NumGraph
    XFeat=FeatureSets{i}; [~,Nx]=size(XFeat);XLab=NodeLabTotal{i};
    DistxxFeat=pdist2(XFeat',XFeat');
    KxxFeat=exp(-(DistxxFeat/sigmaFeat));
    SelfTerm(i)=sum(KxxFeat(XLab==XLab'))/(Nx*Nx);
end

%% Compute the pairwise MMD distance
MMDist=zeros(NumGraph); % squared MMD distance
for i=1:NumGraph
    
    for j=i+1:NumGraph
        XFeat=FeatureSets{i}; [~,Nx]=size(XFeat);XLab=NodeLabTotal{i};
        YFeat=FeatureSets{j}; [~,Ny]=size(YFeat);YLab=NodeLabTotal{j};
        DistxyFeat=pdist2(XFeat',YFeat');
        KxyFeat=exp(-(DistxyFeat/sigmaFeat));
        MMDist(i,j)=SelfTerm(i)+SelfTerm(j)-2*sum(KxyFeat(XLab==YLab'))/(Nx*Ny);
    end
    
end
MMDist=MMDist+MMDist';
MMDist=sqrt(MMDist);
%% Compute the pairwise kernel
sigma2=median(MMDist(:));
K=exp(-MMDist/sigma2);
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
