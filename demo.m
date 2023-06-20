clc; clear; close all; 
% This demo runs the feature selection method VCSDFS on  COIL20 dataset and report the evaluation measures ACC and NMI.
% Please consider Experimental settings for get the better results.
load COIL20;
X=fea;                            % data matrix.
Y=gnd;                            % ground truth lables.
[~,No_of_feat] = size(X);         % Number of features=d.
No_class = length(unique(Y));     % Number of classes.
rho=10^(-3);                      % Balancing parameter. Should be tuned from {10^(-6),....,10^8}, in general.
k=60;                             % Number of selected features. (It can be given by the user)
iter_max=30;                      % Iteration number in proposed algorithm.
[Selected_feat] = VCSDFS(X,rho,k,iter_max);  % Applying VCSDFS feature selecion algorithm.
Kmeans_iter_max=20;               %Since kmeans initiates random we iterate it 20 times and report average values
tempACC=zeros(Kmeans_iter_max,1);
tempNMI=zeros(Kmeans_iter_max,1);
for i=1:Kmeans_iter_max
    IDX = kmeans(X(:,Selected_feat),No_class,'emptyaction','singleton','Replicates',5); %% Running kmeans. Shoud be repeated 20 times, since it relys on random initialization 
    tempACC(i)=100*clusterAcc(Y,IDX);
    tempNMI(i)=100*nmi(Y,IDX);
end
ACC=mean(tempACC);
NMI=mean(tempNMI);
Text=sprintf(' The values of ACC and NMI are %f and %f, respectively.', ACC, NMI);
disp(Text)