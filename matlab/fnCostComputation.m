function [Cost] =  fnCostComputation(x_traj,u_new,p_target)
global a b dt D p c r ucost xcost;


 [~,Horizon] = size(u_new);
 Cost = 0;
 
 for j =1:(Horizon)
     
    %Cost = Cost + exp(-D*dt)*(p*u_new(j)-c*u_new(j)^3/x_traj(j))*dt;
    %quadratic cost
    Cost = Cost + 1/2*(sum(dot(xcost,x_traj(:,j+r).^2))+ucost*sum(u_new(:,j).^2));
    
 end
 
 TerminalCost= 0;
 
 Cost = Cost + TerminalCost;