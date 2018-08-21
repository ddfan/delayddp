function [x] = fnsimulate(x0,u_new,Horizon,sigma)
global dt r;

x = zeros(length(x0),Horizon+r);
x(:,1:r+1)=repmat(x0,1,r+1);

for k = 1:(Horizon-1)
    u_new_noise=[];
    for j=1:size(u_new,1)
        u_new_noise(j)=u_new(j,k)+ u_new(j,k)*sqrt(dt)*sigma*randn;
    end
    
    x(:,k+r+1)=x(:,k+r)+dt*dxdtfunc(x(:,k+r),x(:,k),u_new_noise);
%     x(:,k+r+1)=dxdtfunc(x(:,k+r),x(:,k),u_new_noise);
end