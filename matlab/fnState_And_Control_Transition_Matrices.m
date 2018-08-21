function [f_x,f_u,f_xx,f_uu,f_xu,f_ux] = fnState_And_Control_Transition_Matrices(x,xr,u)
global dt xdim udim r

f_x=zeros(xdim,xdim,r+1);
f_u=zeros(xdim,udim);
f_xx=zeros(xdim,xdim,xdim,r+1,r+1);
f_xu=zeros(xdim,udim,xdim,r+1);
f_uu=zeros(udim,udim,xdim);
f_ux=zeros(udim,xdim,xdim,r+1);

% f_x(:,:,1)=[0 0;x(2) x(1)];
% f_x(:,:,r+1)=[-1 0;0 sin(xr(2))];
% f_u=[1;0];
% f_xx(:,:,2,1,1)=[0 1;1 0];
% f_xx(:,:,2,r+1,r+1)=[0 0; 0 cos(xr(2))];

% two tank cstr, linear dynamics
e=@(z) exp(25*z/(2+z));
de=@(z) exp(25*z/(2+z))*(25*z*(-1/(2+z)^2)+25/(2+z));
f_x(:,:,1)=dt*[-1-e(x(2)), -(x(1)+0.5)*de(x(2)), 0, 0;...
    e(x(2)), -2-u(1)+(x(1)+0.5)*de(x(2)), 0, 0;...
    0, 0, -1-e(x(4)), -(x(3)+0.25)*de(x(4));...
    0, 0, e(x(4)), -2-u(2)+(x(3)+0.25)*de(x(4))];
f_x(:,:,1)=f_x(:,:,1)+eye(4);
f_x(3,1,r+1)=dt;
f_x(4,2,r+1)=dt;
f_u=dt*[0,0;-(x(2)+0.25),0;0,0;0,-(x(4)+0.25)];

f_xu(2,1,2,1)=-dt;
f_xu(4,2,4,1)=-dt;

% % simple toy problem
% f_x(:,:,1)=-sin(x(1))*0;
% f_x(:,:,r+1)=-cos(xr(1))*0;
% f_u=1;
% f_xx(:,:,:,1,1)=-cos(x(1))*0;
% f_xx(:,:,:,r+1,r+1)=sin(xr(1))*0;


% two state toy problem
% f_x(:,:,1)=[0, 1; -2, -1]*dt;
% f_x(:,:,r+1)=[0,0; -10, -5]*dt;
% f_u=dt*[0;1];
% 
% c=0.01; %damping
% m=0.1; %mass
% g=9.8; %gravity
% l=1; %length
% f_x(:,:,r+1)=[0 1; -g/l*cos(xr(1)) -c/m]*dt;
% f_x(:,:,1)=f_x(:,:,1)+eye(2);
% f_u=[0;1]*dt;
% f_xx(:,:,2,1,1)=[g/l*sin(xr(1)) 0; 0 0]*dt;
