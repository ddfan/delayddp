# Differential Dynamic Programming (DDP) with delays
# Author: David Fan
# Date: 11/01/15

import copy
import numpy as np
import pandas as pd
import math
class DDPDelay:
    def __init__(self,params):
        self.sys=params['sys']
        self.loss=params['loss']
        self.Horizon=params['Horizon']
        self.gamma=params['gamma']
        self.min_gamma=params['min_gamma']
        self.gamma_factor=params['gamma_factor']
        self.mu=params['mu']
        self.USE_SECOND_ORDER=params['second_order']
        self.r=params['r']
        self.dt=params['dt']
        self.stopratio=params['stopratio']
        self.discount=params['discount']
        self.num_iter=params['num_iter']

        self.verbose=params['verbose']
        self.schedule=params['schedule']

    def run(self,x0,uk0):
        #initial state/initial control
        self.xtraj=self.sys.runSystem(x0,uk0)
        xold=copy.copy(self.xtraj)
        self.uk=uk0
        
        lx=np.zeros((self.sys.xdim,self.r+1,self.Horizon))
        lu=np.zeros((self.sys.udim,self.Horizon))
        lxu=np.zeros((self.sys.xdim,self.sys.udim,self.r+1,self.Horizon))
        luu=np.zeros((self.sys.udim,self.sys.udim,self.Horizon))
        lxx=np.zeros((self.sys.xdim,self.sys.xdim,self.r+1,self.r+1,self.Horizon))
        fx=np.zeros((self.sys.xdim,self.sys.xdim,self.r+1,self.Horizon))
        fu=np.zeros((self.sys.xdim,self.sys.udim,self.Horizon))
        if self.USE_SECOND_ORDER:
            fxx=np.zeros((self.sys.xdim,self.sys.xdim,self.sys.xdim,self.r+1,self.r+1,self.Horizon))
            fxu=np.zeros((self.sys.xdim,self.sys.udim,self.sys.xdim,self.r+1,self.Horizon))
            fuu=np.zeros((self.sys.udim,self.sys.udim,self.sys.xdim,self.Horizon))
        Vx=np.zeros((self.sys.xdim,self.r+1,self.Horizon+1))
        Vxx=np.zeros((self.sys.xdim,self.sys.xdim,self.r+1,self.r+1,self.Horizon+1))
        Qx=np.zeros((self.sys.xdim,self.r+1))
        Qxu=np.zeros((self.sys.xdim,self.sys.udim,self.r+1))
        Qxx=np.zeros((self.sys.xdim,self.sys.xdim,self.r+1,self.r+1))
        k=np.zeros((self.sys.udim,self.Horizon))
        K=np.zeros((self.sys.udim,self.sys.xdim,self.r+1,self.Horizon))
        
        CONTINUE_ITERATIONS=True
        gamma=self.gamma
        cost=np.zeros(1)
        cost[0]=np.sum(self.loss.runSystemWithSequence(self.xtraj[:,2:], self.uk))
        if self.verbose:
            print('Iteration #:',0,'Cost:',cost[0])
        n=0
        while CONTINUE_ITERATIONS:
            for i in range(self.Horizon):
                lu[:,i]=self.loss.getFu(self.xtraj[:,i:i+self.r+1],self.uk[:,i:i+1])*math.pow(self.discount,self.Horizon-i)
                fu[:,:,i]=self.sys.getFu(self.xtraj[:,i:i+self.r+1],self.uk[:,i:i+1])
                fx[:,:,:,i]=self.sys.getFx(self.xtraj[:,i:i+self.r+1],self.uk[:,i:i+1])
                lx[:,:,i]=np.squeeze(self.loss.getFx(self.xtraj[:,i:i+self.r+1],self.uk[:,i:i+1]))*math.pow(self.discount,self.Horizon-i)
                luu[:,:,i]=np.squeeze(self.loss.getFuu(self.xtraj[:,i:i+self.r+1],self.uk[:,i:i+1]),axis=2)*math.pow(self.discount,self.Horizon-i)
                lxu[:,:,:,i]=np.squeeze(self.loss.getFxu(self.xtraj[:,i:i+self.r+1],self.uk[:,i:i+1]),axis=2)*math.pow(self.discount,self.Horizon-i)
                lxx[:,:,:,:,i]=np.squeeze(self.loss.getFxx(self.xtraj[:,i:i+self.r+1],self.uk[:,i:i+1]),axis=2)*math.pow(self.discount,self.Horizon-i)
                if self.USE_SECOND_ORDER:
                    fuu[:,:,:,i]=self.sys.getFuu(self.xtraj[:,i:i+self.r+1],self.uk[:,i:i+1])
                    fxu[:,:,:,:,i]=self.sys.getFxu(self.xtraj[:,i:i+self.r+1],self.uk[:,i:i+1])
                    fxx[:,:,:,:,:,i]=self.sys.getFxx(self.xtraj[:,i:i+self.r+1],self.uk[:,i:i+1])

            for j in range(self.r+1):
                Vx[:,j,self.Horizon]=lx[:,j,-1]
                for l in range(self.r+1):
                    Vxx[:,:,j,l,self.Horizon]=lxx[:,:,j,l,-1]
        
            for i in range(self.Horizon-1,-1,-1):
#                 print 'Vxx:',np.amin(np.linalg.eigvalsh(Vxx[:,:,0,0,i+1])),
                if self.USE_SECOND_ORDER:
                    w, v=np.linalg.eigh(Vxx[:,:,0,0,i+1])
                    w=w+self.mu
                    Vxx[:,:,0,0,i+1]=np.dot(v,np.dot(np.diag(w),v.T))
#                     print np.amin(np.linalg.eigvalsh(Vxx[:,:,0,0,i+1]))

                Qu=lu[:,i]+np.dot(fu[:,:,i].T,Vx[:,0,i+1])
                Quu=luu[:,:,i]+np.dot(fu[:,:,i].T,np.dot(Vxx[:,:,0,0,i+1],fu[:,:,i]))
                if self.USE_SECOND_ORDER:
                    Quu+=np.dot(fuu[:,:,:,i],Vx[:,0,i+1])
                    
#                 print 'Quu:',np.amin(np.linalg.eigvalsh(Quu))
                
                for j in range(self.r+1):
                    Qx[:,j]=lx[:,j,i]+np.dot(fx[:,:,j,i].T,Vx[:,0,i+1])
                    if j<self.r:
                        Qx[:,j]+=Vx[:,j+1,i+1]
                    Qxu[:,:,j]=lxu[:,:,j,i]+np.dot(fx[:,:,j,i].T,np.dot(Vxx[:,:,0,0,i+1],fu[:,:,i]))
                    if self.USE_SECOND_ORDER:
                        Qxu[:,:,j]+=np.dot(fxu[:,:,:,j,i],Vx[:,0,i+1])
                    if j<self.r:
                        Qxu[:,:,j]+=np.dot(Vxx[:,:,j+1,0,i+1],fu[:,:,i])
                    for l in range(self.r+1):
                        Qxx[:,:,j,l]=lxx[:,:,j,l,i]+np.dot(fx[:,:,j,i].T,np.dot(Vxx[:,:,0,0,i+1],fx[:,:,l,i]))
                        if self.USE_SECOND_ORDER:
                            Qxx[:,:,j,l]+=np.dot(fxx[:,:,:,j,l,i],Vx[:,0,i+1])
                        if j<self.r:
                            Qxx[:,:,j,l]+=np.dot(Vxx[:,:,j+1,0,i+1],fx[:,:,l,i])
                        if l<self.r:
                            Qxx[:,:,j,l]+=np.dot(fx[:,:,j,i].T,Vxx[:,:,0,l+1,i+1])
                        if j<self.r and l<self.r:
                            Qxx[:,:,j,l]+=Vxx[:,:,j+1,l+1,i+1]
                
                k[:,i]=-np.linalg.solve(Quu,Qu)
                for j in range(self.r+1):
                    K[:,:,j,i]=-np.linalg.solve(Quu,Qxu[:,:,j].T)
                    Vx[:,j,i]=Qx[:,j]+np.dot(K[:,:,j,i].T,np.dot(Quu,k[:,i]))+np.dot(K[:,:,j,i].T,Qu)+np.dot(Qxu[:,:,j],k[:,i])
                    for l in range(self.r+1):
                        Vxx[:,:,j,l,i]=Qxx[:,:,j,l]+np.dot(K[:,:,j,i].T,np.dot(Quu,K[:,:,l,i]))+np.dot(Qxu[:,:,j],K[:,:,l,i])+np.dot(K[:,:,j,i].T,Qxu[:,:,l].T)
                
#             #find controls
#             while True:
#                 uk_new=self.uk
#                 xtraj_new=self.xtraj
#                 print gamma
#                 for i in xrange(0,self.Horizon):
#                     uk_new[:,i:i+1] += gamma*k[:,i:i+1]
#                     for j in xrange(self.r+1):
#                         uk_new[:,i:i+1] += np.dot(K[:,:,j,i], self.xtraj[:,i+self.r-j:i+self.r-j+1]-xold[:,i+self.r-j:i+self.r-j+1])
#                     xtraj_new[:,i+self.r+1:i+self.r+2]=self.sys.getNextState(xtraj_new[:,i:i+self.r+1],self.uk[:,i:i+1])
#                 
#                 newcost=np.sum(self.loss.runSystemWithSequence(self.xtraj[:,2:], self.uk))
#                 if cost[-1] < newcost:
#                     if gamma/self.gamma_factor < self.min_gamma:
#                         CONTINUE_ITERATIONS=False
#                         break
#                     gamma=gamma/self.gamma_factor
#                 else:
#                     n=n+1
#                     cost=np.append(cost,newcost)
#                     self.xtraj=xtraj_new
#                     self.uk=uk_new
#                     xold=copy.copy(self.xtraj)
#                     if self.verbose:
#                         print 'Iteration #:',n,'Cost:',cost[-1]
# #                     gamma=gamma*self.gamma_factor
#                         
#                     if cost[-2]-cost[-1] < self.stopratio:
#                         CONTINUE_ITERATIONS=False
#                     break
                #find controls
            uk_new=self.uk
            xtraj_new=self.xtraj
            if n >= len(self.schedule):
                gamma=self.schedule[-1]
            else:
                gamma=self.schedule[n]
        
            print(gamma)
            for i in range(0,self.Horizon):
                uk_new[:,i:i+1] += gamma*k[:,i:i+1]
                for j in range(self.r+1):
                    uk_new[:,i:i+1] += np.dot(K[:,:,j,i], self.xtraj[:,i+self.r-j:i+self.r-j+1]-xold[:,i+self.r-j:i+self.r-j+1])
                xtraj_new[:,i+self.r+1:i+self.r+2]=self.sys.getNextState(xtraj_new[:,i:i+self.r+1],self.uk[:,i:i+1])
            
            newcost=np.sum(self.loss.runSystemWithSequence(self.xtraj[:,2:], self.uk))
            
            if cost[-1]<newcost:
                break
            
            n=n+1
            cost=np.append(cost,newcost)
            self.xtraj=xtraj_new
            self.uk=uk_new
            xold=copy.copy(self.xtraj)
            if self.verbose:
                print('Iteration #:',n,'Cost:',cost[-1])
#                     gamma=gamma*self.gamma_factor

            if cost[-2]-cost[-1] < self.stopratio or n>self.num_iter:
                CONTINUE_ITERATIONS=False

            self.K=K
        

