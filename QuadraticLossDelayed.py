# a pendulum class.  inherits DelayedDiscreteSystem class.

from DelayedDiscreteSystem import DelayedDiscreteSystem

import theano.tensor as T
import math

class QuadraticLossDelayed(DelayedDiscreteSystem):
    def __init__(self,params):
        DelayedDiscreteSystem.__init__(self)
        self.dt=params['dt']
        self.xdim=params['xdim']
        self.udim=params['udim']
        self.xcost=params['xcost']
        self.ucost=params['ucost']
        self.xT=params['xT']
        self.r=params['r']
        
        x=self.xu_flat[-(self.xdim+self.udim):-self.udim].dimshuffle(0,'x')
        u=self.xu_flat[-self.udim:].dimshuffle(0,'x')
        
        self.xnew=0.5*(T.dot(self.xcost.T,(x-self.xT)*(x-self.xT))+T.dot(self.ucost.T,u*u))
        
        DelayedDiscreteSystem.init_theano_vars(self)

class QuadraticLossDelayedDifference(DelayedDiscreteSystem):
    def __init__(self,params):
        DelayedDiscreteSystem.__init__(self)
        self.dt=params['dt']
        self.xdim=params['xdim']
        self.udim=params['udim']
        self.xcost=params['xcost']
        self.ucost=params['ucost']
        self.xT=params['xT']
        self.r=params['r']
        self.diffcost=params['diffcost']
        
        x=self.xu_flat[-(self.xdim+self.udim):-self.udim].dimshuffle(0,'x')
        xm1=self.xu_flat[-(self.xdim*2+self.udim):-(self.xdim+self.udim)].dimshuffle(0,'x')
        u=self.xu_flat[-self.udim:].dimshuffle(0,'x')
        
        self.xnew=0.5*(T.dot(self.xcost.T,(x-self.xT)*(x-self.xT))+T.dot(self.ucost.T,u*u)+T.dot(self.diffcost.T,(x-xm1)*(x-xm1)))
        
        DelayedDiscreteSystem.init_theano_vars(self)
        
class QuadraticLossDelayedAnglePendulum(DelayedDiscreteSystem):
    def __init__(self,params):
        DelayedDiscreteSystem.__init__(self)
        self.dt=params['dt']
        self.xdim=params['xdim']
        self.udim=params['udim']
        self.xcost=params['xcost']
        self.ucost=params['ucost']
        self.xT=params['xT']
        self.r=params['r']
        self.diffcost=params['diffcost']
        
        x=self.xu_flat[-(self.xdim+self.udim):-self.udim]
        xm1=self.xu_flat[-(self.xdim*2+self.udim):-(self.xdim+self.udim)].dimshuffle(0,'x')
        u=self.xu_flat[-self.udim:].dimshuffle(0,'x')
        theta=(T.arctan2(x[1],x[0])+math.pi/2.0).dimshuffle('x').dimshuffle(0,'x')
        thetam1=(T.arctan2(xm1[1],xm1[0])+math.pi/2.0).dimshuffle('x').dimshuffle(0,'x')
        self.xnew=0.5*(T.dot(self.xcost.T,(theta-self.xT)*(theta-self.xT))+T.dot(self.ucost.T,u*u)+T.dot(self.diffcost.T,(theta-thetam1)*(theta-thetam1)))
#         self.xnew=0.5*(T.dot(self.xcost.T,(theta-self.xT)*(theta-self.xT))+T.dot(self.ucost.T,u*u))
        
        DelayedDiscreteSystem.init_theano_vars(self)
        
class QuadraticLossDelayedAngleDoubleCartPole(DelayedDiscreteSystem):
    def __init__(self,params):
        DelayedDiscreteSystem.__init__(self)
        self.dt=params['dt']
        self.xdim=params['xdim']
        self.udim=params['udim']
        self.xcost=params['xcost']
        self.ucost=params['ucost']
        self.xT=params['xT']
        self.r=params['r']
        self.diffcost=params['diffcost']
        
        x=self.xu_flat[-(self.xdim+self.udim):-self.udim]
        xm1=self.xu_flat[-(self.xdim*2+self.udim):-(self.xdim+self.udim)].dimshuffle(0,'x')
        u=self.xu_flat[-self.udim:].dimshuffle(0,'x')
        theta1=(T.arctan2(x[3],x[1])+math.pi/2.0).dimshuffle('x').dimshuffle(0,'x')
        theta2=(T.arctan2(x[4],x[2])+math.pi/2.0).dimshuffle('x').dimshuffle(0,'x')
        theta1m1=(T.arctan2(xm1[3],xm1[1])+math.pi/2.0).dimshuffle('x').dimshuffle(0,'x')
        theta2m1=(T.arctan2(xm1[4],xm1[2])+math.pi/2.0).dimshuffle('x').dimshuffle(0,'x')
        theta=T.concatenate([theta1,theta2],axis=1)
        thetam1=T.concatenate([theta1m1,theta2m1],axis=1)
        self.xnew=T.sum(0.5*(T.dot(self.xcost.T,(theta-self.xT)*(theta-self.xT))+T.dot(self.ucost.T,u*u)+T.dot(self.diffcost.T,(theta-thetam1)*(theta-thetam1))))
        
        DelayedDiscreteSystem.init_theano_vars(self)
        
class QuadraticLossDelayedAngleDoubleCartPolePosVel(DelayedDiscreteSystem):
    def __init__(self,params):
        DelayedDiscreteSystem.__init__(self)
        self.dt=params['dt']
        self.xdim=params['xdim']
        self.udim=params['udim']
        self.xcost=params['xcost']
        self.ucost=params['ucost']
        self.xT=params['xT']
        self.r=params['r']
        
        x=self.xu_flat[-(self.xdim+self.udim):-self.udim]
        u=self.xu_flat[-self.udim:].dimshuffle(0,'x')
        xjoined=T.stack([x[0],T.arctan2(x[6],x[1])+math.pi/2.0,T.arctan2(x[7],x[2])+math.pi/2.0,x[3],x[4],x[5]]).dimshuffle(0,'x')
        self.xnew=T.sum(0.5*(T.dot(self.xcost.T,(xjoined-self.xT)*(xjoined-self.xT))+T.dot(self.ucost.T,u*u)))
        
        DelayedDiscreteSystem.init_theano_vars(self)

class QuadraticLossDelayedAngleSingleCartPole(DelayedDiscreteSystem):
    def __init__(self,params):
        DelayedDiscreteSystem.__init__(self)
        self.dt=params['dt']
        self.xdim=params['xdim']
        self.udim=params['udim']
        self.xcost=params['xcost']
        self.ucost=params['ucost']
        self.xT=params['xT']
        self.r=params['r']
        
        x=self.xu_flat[-(self.xdim+self.udim):-self.udim]
        u=self.xu_flat[-self.udim:].dimshuffle(0,'x')
        xjoined=T.stack([x[0],T.arctan2(x[2],x[1])+math.pi/2.0]).dimshuffle(0,'x')
        self.xnew=T.sum(0.5*(T.dot(self.xcost.T,(xjoined-self.xT)*(xjoined-self.xT))+T.dot(self.ucost.T,u*u)))
        
        DelayedDiscreteSystem.init_theano_vars(self)
        
        
class QuadraticLossDelayedAngleSingleCartPolePosVel(DelayedDiscreteSystem):
    def __init__(self,params):
        DelayedDiscreteSystem.__init__(self)
        self.dt=params['dt']
        self.xdim=params['xdim']
        self.udim=params['udim']
        self.xcost=params['xcost']
        self.ucost=params['ucost']
        self.xT=params['xT']
        self.r=params['r']
        
        x=self.xu_flat[-(self.xdim+self.udim):-self.udim]
        u=self.xu_flat[-self.udim:].dimshuffle(0,'x')
        xjoined=T.stack([x[0],T.arctan2(x[4],x[1])+math.pi/2.0,x[2],x[3]]).dimshuffle(0,'x')
        self.xnew=T.sum(0.5*(T.dot(self.xcost.T,(xjoined-self.xT)*(xjoined-self.xT))+T.dot(self.ucost.T,u*u)))
        
        DelayedDiscreteSystem.init_theano_vars(self)
        
class QuadraticLossDelayedAngleDoublePendulum(DelayedDiscreteSystem):
    def __init__(self,params):
        DelayedDiscreteSystem.__init__(self)
        self.dt=params['dt']
        self.xdim=params['xdim']
        self.udim=params['udim']
        self.xcost=params['xcost']
        self.ucost=params['ucost']
        self.xT=params['xT']
        self.r=params['r']
        self.diffcost=params['diffcost']
        
        x=self.xu_flat[-(self.xdim+self.udim):-self.udim]
        xm1=self.xu_flat[-(self.xdim*2+self.udim):-(self.xdim+self.udim)].dimshuffle(0,'x')
        u=self.xu_flat[-self.udim:].dimshuffle(0,'x')
        theta1=(T.arctan2(x[1],x[0])+math.pi/2.0).dimshuffle('x').dimshuffle(0,'x')
        theta2=(T.arctan2(x[3],x[2])+math.pi/2.0).dimshuffle('x').dimshuffle(0,'x')
        theta1m1=(T.arctan2(xm1[1],xm1[0])+math.pi/2.0).dimshuffle('x').dimshuffle(0,'x')
        theta2m1=(T.arctan2(xm1[3],xm1[2])+math.pi/2.0).dimshuffle('x').dimshuffle(0,'x')
        theta=T.concatenate([theta1,theta2],axis=1)
        thetam1=T.concatenate([theta1m1,theta2m1],axis=1)
        self.xnew=T.sum(0.5*(T.dot(self.xcost.T,(theta-self.xT)*(theta-self.xT))+T.dot(self.ucost.T,u*u)+T.dot(self.diffcost.T,(theta-thetam1)*(theta-thetam1))))
        
        DelayedDiscreteSystem.init_theano_vars(self)