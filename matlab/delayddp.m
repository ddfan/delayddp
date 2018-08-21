%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%  Iterative Linear Quadratic Regulator for two Link Arm Rigid Body Dynamics       %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%  Course: Advance Topics on Stochastic Optimal Control and Reinforcement Learning %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%  AE8803 Spring 2014                             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%  Author: Evangelos Theodorou                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Modified by David Fan to create delay ddp on 09/09/2015


clear all;

global dt r ucost xcost xdim udim;
% Horizon
Horizon = 101;
% Discretization
dt = 0.05;
% Delay (num timesteps)
r = 10;
%initial condition
x0=[0.15;-0.03;0.1;0];
%control dim
udim=2;
xdim=length(x0);
%scale initial noisy control input
uscale=0.0;
%noise on input
sigma=0.0;
%scale input cost
ucost=0.01;
xcost=[1 1 1 1];
xcostfinal=[1 1 1 1];

% % Learning Rate:
% gamma0 = 0.1;
% gamma_min=1e-10;
% %regularize control
% mu=0.0;

% Number of Iterations
num_iter = 100;
% Learning Rate:
gamma = 0.4;
%regularize control
mu=0.0;

rng(12)
% Initial Control:
% load('u_init')
u_k = uscale*randn(udim,Horizon-1);
du_k = zeros(udim,Horizon-1);
% Initial trajectory:
x_traj = fnsimulate(x0,u_k,Horizon,sigma);
% x_traj2 = fnsimulate(x0,u_k,Horizon,sigma);
%no control
x_init = fnsimulate(x0,0*u_k,Horizon,0);
x_old=x_traj;
Cost=0;
% NOT_CONVERGED=true;
% n=1;
% while NOT_CONVERGED
[Cost(:,1)] =  fnCostComputation(x_traj,u_k);
fprintf('Iteration %d,  Current Cost = %e \n',0,Cost(1,1));

for n = 1:num_iter
    %------------------------------------------------> Quadratic Approximations of the cost function
    for i=1:Horizon-1
        [l0(i),l_x(:,:,i),l_xx(:,:,:,:,i),l_u(:,i),l_uu(:,:,i),l_xu(:,:,:,i)] = fnCost(x_traj(:,i+r), x_traj(:,i), u_k(:,i));
        [f_x(:,:,:,i),f_u(:,:,i),f_xx(:,:,:,:,:,i),f_uu(:,:,:,i),f_xu(:,:,:,:,i)] = fnState_And_Control_Transition_Matrices(x_traj(:,i+r),x_traj(:,i),u_k(:,i));
    end
    
    %------------------------------------------------> Find the last value
    Vx=zeros(xdim,r+1,Horizon);
    Vxx=zeros(xdim,xdim,r+1,r+1,Horizon);
    for j=1:1%r+1
        Vx(:,j,Horizon) = xcostfinal'.*x_traj(:,end);
        for l=1:1%r+1
            Vxx(:,:,j,l,Horizon)= diag(xcostfinal);
        end
    end
    
    %------------------------------------------------> Backpropagation of the Value Function
    for i = (Horizon-1):-1:1
        Qu = l_u(:,i)+f_u(:,:,i)'*Vx(:,1,i+1);
        Quu  = l_uu(:,:,i)+f_u(:,:,i)'*(Vxx(:,:,1,1,i+1))*f_u(:,:,i);%+tensorsum(Vx(:,1,i+1),f_uu(:,:,:,i));
        for j=1:r+1
            %-------Update Qx--------
            Qx(:,j)  = l_x(:,j,i) + f_x(:,:,j,i)'*Vx(:,1,i+1);
            if j<r+1
                Qx(:,j) = Qx(:,j) + Vx(:,j+1,i+1);
            end
            %------------------------
            
            %-------Update Qxu--------
            Qxu(:,:,j)=l_xu(:,:,j,i) + f_x(:,:,j,i)'*Vxx(:,:,1,1,i+1)*f_u(:,:,i);
            if j<r+1
                Qxu(:,:,j) = Qxu(:,:,j) + Vxx(:,:,j+1,1,i+1)*f_u(:,:,i);
            end
%             Qxu(:,:,j)=Qxu(:,:,j) + tensorsum(Vx(:,1,i+1),f_xu(:,:,:,j,i));
            %-------------------------
            
            %-------Update Qxx--------
            for l=1:r+1
                Qxx(:,:,j,l) = l_xx(:,:,j,l,i) + f_x(:,:,j,i)'*Vxx(:,:,1,1,i+1)*f_x(:,:,l,i);
                if j<r+1
                    Qxx(:,:,j,l)=Qxx(:,:,j,l) + Vxx(:,:,j+1,1,i+1)*f_x(:,:,l,i);
                end
                if l<r+1
                    Qxx(:,:,j,l)=Qxx(:,:,j,l) + f_x(:,:,j,i)'*Vxx(:,:,1,l+1,i+1);
                end
                if j<r+1 && l<r+1
                    Qxx(:,:,j,l)=Qxx(:,:,j,l) + Vxx(:,:,j+1,l+1,i+1);
                end
%                 if l>=p+1
%                     Qxx(:,:,j,l)=Qxx(:,:,j,l) + tensorsum(Vx(:,p+1,i+1),f_xx(:,:,:,j-p,l-p,i-p));
%                 end
            end
            %------------------------
        end
        
        %regularize
        if min(eig(Quu)) <= 0
            Quu=Quu+(mu+abs(min(eig(Quu))))*eye(size(Quu));
        end
%         Quu=Quu+mu*eye(udim);
        invQuu=inv(Quu);
        k(:,i)= -invQuu*Qu;
        for j=1:r+1
            K(:,:,j,i) = -invQuu*(Qxu(:,:,j)');
        end
        
        delV1(i) = k(:,i)'*Qu;
        delV2(i) = 1/2*k(:,i)'*Quu*k(:,i);
        for j=1:r+1
            Vx(:,j,i) = Qx(:,j)+K(:,:,j,i)'*Quu*k(:,i)+K(:,:,j,i)'*Qu+Qxu(:,:,j)*k(:,i);
            for l=1:r+1
                Vxx(:,:,j,l,i) = Qxx(:,:,j,l)+K(:,:,j,i)'*Quu*K(:,:,l,i)+Qxu(:,:,j)*K(:,:,l,i)+K(:,:,j,i)'*Qxu(:,:,l)';
            end
        end
    end
    
    
    %     %---------------FORWARD PASS -----------------------------
    %     gamma=gamma0;
    %     cost =  fnCostComputation(x_traj,u_k);
    %
    %     while true
    %         u_k_new=u_k;
    %         x_traj_new=x_traj;
    %         %----------------------------------------------> Find the controls
    %         for i=1:(Horizon-1)
    %             du=gamma * k(:,i);
    %             for j=1:r+1
    %                 du = du + K(:,:,j,i)*(x_traj_new(:,i+r-(j-1))-x_old(:,i+r-(j-1)));
    %             end
    %
    %             u_k_new(:,i) = u_k(:,i) + du;
    %
    %             u_new_noise=[];
    %             for j=1:size(u_k_new,1)
    %                 u_new_noise(j,i)=u_k_new(j,i)+ u_k_new(j,i)*sqrt(dt)*sigma*randn;
    %             end
    %
    %             x_traj_new(:,i+r+1)=x_traj_new(:,i+r)+dt*dxdtfunc(x_traj_new(:,i+r),x_traj_new(:,i),u_new_noise(:,i));
    %             %         x_traj(:,i+r+1)=dxdtfunc(x_traj(:,i+r),x_traj(:,i),u_k(:,i));
    %         end
    %
    %
    %         %---------------------------------------------> Simulation of the Nonlinear System
    %         cost_new =  fnCostComputation(x_traj_new,u_k_new);
    %         if (cost-cost_new < 0 || isnan(cost_new))
    %             if gamma < gamma_min
    %                 NOT_CONVERGED=false;
    %                 break;
    %             end
    %             gamma=gamma/1.5;
    %             continue;
    %         elseif  (length(Cost)>1 && abs(Cost(end)-Cost(end-1)) <= 1e-6)
    %             NOT_CONVERGED=false;
    %             break;
    %         else
    %             u_k=u_k_new;
    %             x_traj=x_traj_new;
    %             x_old=x_traj_new;
    %             Cost(n)=cost_new;
    %             fprintf('Iteration %d,  Current Cost = %e \n',n,Cost(n));
    %             n=n+1;
    %             break;
    %         end
    %     end
    
    
    %----------------------------------------------> Find the controls
    for i=1:(Horizon-1)
        du=gamma * k(:,i);
        for j=1:r+1
            du = du + K(:,:,j,i)*(x_traj(:,i+r-(j-1))-x_old(:,i+r-(j-1)));
        end
        u_k(:,i) = u_k(:,i) + du;
        x_traj(:,i+r+1)=x_traj(:,i+r)+dt*dxdtfunc(x_traj(:,i+r),x_traj(:,i),u_k(:,i));
        %         x_traj(:,i+r+1)=dxdtfunc(x_traj(:,i+r),x_traj(:,i),u_k(:,i));
    end
    x_old=x_traj;
    
    %---------------------------------------------> Simulation of the Nonlinear System
    [Cost(:,n+1)] =  fnCostComputation(x_traj,u_k);
    
    fprintf('Iteration %d,  Current Cost = %e \n',n,Cost(1,n+1));
end


%% Plot stuff
time=linspace(0,dt*(Horizon+r),Horizon+r);

figure;
subplot(1,2,1)
plot(time,x_init','linewidth',2)
title('Without Control','fontsize',10);
xlabel('Time(s)','fontsize',10)
legend('x1','x2','x3','x4','location','northwest')
set(gca,'xlim',[0 5.5])
% axis([0 dt*(Horizon+r) -0.3 0.7])
%
subplot(1,2,2)
plot(time,x_traj','linewidth',2);
title('With Control','fontsize',10);
xlabel('Time(s)','fontsize',10)
legend('x1','x2','x3','x4','location','northeast')
% axis([0 dt*(Horizon+r) -0.3 0.7])
set(gca,'xlim',[0 5.5])

figure;
subplot(1,2,1)
plot(time(r+1:length(u_k)+r),u_k,'linewidth',2);
xlabel('Time(s)','fontsize',10)
title('Control','fontsize',10);
% set(gca,'xlim',[0 dt*(Horizon+r)])
legend('u1','u2','location','northeast')
set(gca,'xlim',[0 5.5])
set(gca,'ylim',[-1 0.6])

subplot(1,2,2)
plot(1:Horizon,Cost,'linewidth',2);
xlabel('Iterations','fontsize',10)
title('Cost','fontsize',10);
set(gca,'xlim',[0 20])
% 
% figure;
% subplot(1,2,1)
% plot(time(1:101),x_traj2','linewidth',2)
% title('System Without Delay','fontsize',10);
% xlabel('Time(s)','fontsize',10)
% legend('x1','x2','x3','x4','location','northeast')
% set(gca,'xlim',[0 5.5])
% set(gca,'ylim',[-0.07 0.16])
% 
% % axis([0 dt*(Horizon+r) -0.3 0.7])
% %
% subplot(1,2,2)
% plot(time,x_traj','linewidth',2);
% title('System With Delay','fontsize',10);
% xlabel('Time(s)','fontsize',10)
% legend('x1','x2','x3','x4','location','northeast')
% % axis([0 dt*(Horizon+r) -0.3 0.7])
% set(gca,'ylim',[-0.07 0.16])
% set(gca,'xlim',[0 5.5])

%save('DDP_Data');
save('u_init','u_k')