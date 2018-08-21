# a double cart and pole class.  inherits DiscreteSystem class.

from DiscreteSystem import DiscreteSystem

import theano.tensor as T
import numpy as np
import math
import copy

class SingleCartPole(DiscreteSystem):
    def __init__(self,params):
        DiscreteSystem.__init__(self)
        
        self.dt=params['dt']
        self.xdim=4
        self.udim=1
        
        M=params['M']
        m=params['m']
        L=params['L']
        g=params['g']
        k1=params['k1'] #joint friction
        k2=params['k2'] #cart friction
        
        if 'row_sums' in params:
            self.row_sums=params['row_sums']
        if 'row_std' in params:
            self.row_std=params['row_std']
        
        
        dx=T.stack([self.xu[2],
                    self.xu[3],
                    (-m*g*T.sin(self.xu[1])*T.cos(self.xu[1])+m*L*self.xu[3]*self.xu[3]*T.sin(self.xu[1])+k1*m*self.xu[3]*T.cos(self.xu[1])+self.u[0]-k2*self.xu[2])/(M+(1-T.cos(self.xu[1])*T.cos(self.xu[1]))*m),
                    ((M+m)*(g*T.sin(self.xu[1])-k1*self.xu[3])-(L*m*self.xu[3]*self.xu[3]*T.sin(self.xu[1])+self.u[0]-k2*self.xu[2])*T.cos(self.xu[1]))/L/(M+(1.0-T.cos(self.xu[1])*T.cos(self.xu[1]))*m)]).dimshuffle(0,'x')
        
        self.xnew = self.x + self.dt*dx
        
        DiscreteSystem.init_theano_vars(self)
        
    def splitAngle(self,x,posvel=False):
        xnew=copy.copy(x)
        if posvel:
            xnew=np.vstack([xnew,np.zeros((1,xnew.shape[1]))])
            xnew[4,:]=-np.cos(x[1,:])
            xnew[1,:]=np.sin(x[1,:])
            xnew=(xnew-self.row_sums[0,:,:5].T)/self.row_std[0,:,:5].T
        else:
            xnew[2,:]=-np.cos(x[1,:])
            xnew[1,:]=np.sin(x[1,:])
            xnew=np.delete(xnew,3,0)
            xnew=(xnew-self.row_sums[0,:,:3].T)/self.row_std[0,:,:3].T
        
        return xnew

    def joinAngle(self,x):
        xnew=np.zeros_like(x)
        xnew[1,:]=np.arctan2(x[5,:],x[1,:])+math.pi/2.0
        return xnew

