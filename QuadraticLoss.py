# a pendulum class.  inherits DiscreteSystem class.

from DiscreteSystem import DiscreteSystem

import theano.tensor as T

class QuadraticLoss(DiscreteSystem):
    def __init__(self,params):
        DiscreteSystem.__init__(self)
        self.dt=params['dt']
        self.xdim=params['xdim']
        self.udim=params['udim']
        self.xcost=params['xcost']
        self.ucost=params['ucost']
        self.xT=params['xT']
        
        self.xnew=0.5*self.dt*(T.dot(self.xcost.T,T.sqr(self.xu[:self.xdim,:]-self.xT))+T.dot(self.ucost.T,T.sqr(self.xu[-self.udim:,:])))
        
        DiscreteSystem.init_theano_vars(self)
