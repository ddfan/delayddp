function dx = dxdtfunc(x,xr,u)
global dt
dx=x*0;

%optimal fishing problem
% dx(1)=-xr(1)+u;
% dx(2)=-cos(xr(2))+x(1)*x(2);

% two stage cstr
R1=(x(1)+0.5)*exp(25*x(2)/(2+x(2)));
R2=(x(3)+0.25)*exp(25*x(4)/(2+x(4)));
dx(1)=0.5-x(1)-R1;
dx(2)=-2*(x(2)+0.25)-u(1)*(x(2)+0.25)+R1;
dx(3)=xr(1)-x(3)-R2+0.25;
dx(4)=xr(2)-2*x(4)-u(2)*(x(4)+0.25)+R2-0.25;

% simple toy problem
% dx(1)=cos(x)-sin(xr)+u;

%two state toy problem
% dx(1)=x(2);
% dx(2)=(-2*x(1)-1*x(2))*0-10*xr(1)-5*xr(2)+u;

%pendulum
% c=0.01; %damping
% m=0.1; %mass
% g=9.8; %gravity
% l=1; %length
% dx(1)=xr(2);
% dx(2)=-c/m*xr(2)-g/l*sin(xr(1))+u;