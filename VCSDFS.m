function [selected]=VCSDFS(X,rho,k,itermax)
[~,d]=size(X);
W=rand(d,k);
XX=(X')*X;
XXX=(XX)^2;
iter=1;
while iter~=itermax
SW=XXX*W+rho*W;
A=XX*W;
FW=A*(W')*A+rho*ones(d,d)*W;
re1=rdivide(SW,FW);
re1=re1.^(1/4);
W=W.*re1;
iter=iter+1;
end
tempVector = sum(W.^2, 2);
[~, value] = sort(tempVector, 'descend'); % sort tempVecror (W) in a descend order
selected = value(1:k);
end


