function [l0,l_x,l_xx,l_u,l_uu,l_xu,l_ux] = fnCost(x, xr, u)
global dt r ucost xcost xdim udim;

l0=zeros(1);
l_x=zeros(xdim,r+1);
l_xx=zeros(xdim,xdim,r+1,r+1);
l_u=zeros(udim,1);
l_uu=zeros(udim,udim);
l_xu=zeros(xdim,udim,r+1);
l_ux=zeros(udim,xdim,r+1);

% optimal fishing loss fcn
% l0 = (D*dt-log(p*u-c*u^3/x))*dt;
% l_x = -1/(p*x^2/(c*u^2)-x)*dt;
% l_xx = 1/(x^2*(p*x/(c*u^2)-1))*dt;
% l_u = -1/(p*u-c*u^3/x)*(p-3*c*u^2/x)*dt;
% l_uu = (-1/(p*u-c*u^3/x)*(-6*c*u/x)+(p-3*c*u^2/x)/(p*u-c*u^3/x)^2*(-3*c*u^2/x))*dt;
% l_ux = -2*p*x^2/c*u^3/(p*x^2/(c*u^2)-x)^2*dt;

% two stage cstr loss (quadratic)
l0=(1/2*dot(xcost,x.^2)+ucost*1/2*sum(u.^2));
for j=1:1%r+1
    l_x(:,j)=xcost'.*x;
%       for l=1:r+1
        l_xx(:,:,j,j)=diag(xcost);
%       end
end

% l_x(:,r+1)=xcost'.*x;
% l_xx(:,:,r+1,r+1)=diag(xcost);

l_u=ucost*u;
l_uu=ucost*eye(length(u));