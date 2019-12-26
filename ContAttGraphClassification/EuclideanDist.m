function [Dist] = EuclideanDist(X,Y)
% This function is used to compute the pairwise squared Euclidean distances
% between data matrices X and Y
% each column of X(or Y) is an observation
% The output Dist is an Nx*Ny matrices;

% EuclideanDist(X,Y)=pdist2(X',Y').^2;

[~,Nx]=size(X);[~,Ny]=size(Y);
SquaredNormX=sum((X.*X)); % Sum(A) is a row vector, storing the sum of each column of A
SquaredNormY=sum((Y.*Y));
% SquaredNormX and SquaredNormY are 1*Nx and 1*Ny vectors, respectively
Dist=SquaredNormX'*ones(1,Ny)+ones(Nx,1)*SquaredNormY-2*X'*Y;
end

