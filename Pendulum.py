# a pendulum class.  inherits DiscreteSystem class.

from DiscreteSystem import DiscreteSystem

import theano.tensor as T
import numpy as np
import math

class Pendulum(DiscreteSystem):
    def __init__(self,params):
        DiscreteSystem.__init__(self)
        
        self.dt=params['dt']
        self.xdim=2
        self.udim=1
        
        c=params['c']
        m=params['m']
        g=params['g']
        l=params['l']
    
        self.xnew=self.x+self.dt*T.stack([self.xu[1], -c/m*self.xu[1]-g/l*T.sin(self.xu[0])+self.xu[2]])
        
        DiscreteSystem.init_theano_vars(self)
        
    def splitAngle(self,x):
        xnew=np.zeros_like(x)
        xnew[1,:]=-np.cos(x[0,:])
        xnew[0,:]=np.sin(x[0,:])
        return xnew
    
    def joinAngle(self,x):
        xnew=np.zeros_like(x)
        xnew[0,:]=np.arctan2(x[1,:],x[0,:])+math.pi/2.0
        return xnew
