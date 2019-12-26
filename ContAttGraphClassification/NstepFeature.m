function [FeaturesTotal,NodeLab,NodeAtt] = NstepFeature(StepN,G,NodeLab_Exist,NodeAtt_Exist)
% This function is used to compute N-step return probabilities of random
% walks on graphs with labeled nodes. We simply use the mutiplication of
% Probability transition matrices.

NumNodes=G.n_nodes;
FeaturesTotal=zeros(StepN,NumNodes);
AdjMat=G.A+eye(NumNodes); Deg=AdjMat*ones(NumNodes,1); 

if StepN<=5
    InvDegCol=1./Deg; TranProb=sparse(InvDegCol.*AdjMat);
    iTranProb=TranProb;
    for i=1:StepN
        FeaturesTotal(i,:)=diag(iTranProb);
        iTranProb=iTranProb*TranProb;
    end
else
    InvSqrtDeg=1./sqrt(Deg);SysP=InvSqrtDeg.*AdjMat.*(InvSqrtDeg)';SysP=(SysP+SysP')/2;
    [U,D]=eig(SysP);ColD=diag(D);TempD=ColD;
    U=U.*U;
    for s=1:StepN
        FeaturesTotal(s,:)=U*(TempD);
        TempD=TempD.*ColD;
    end
end

%% Label information
if NodeLab_Exist==1
    NodeLab=G.node_labels;
else
    NodeLab=[];
end
%% Attribute information 
if NodeAtt_Exist==1
    NodeAtt=G.node_attributes'; % now each column is an observation 
else
    NodeAtt=[];
end

end

