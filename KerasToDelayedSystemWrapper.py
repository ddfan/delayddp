from DelayedDiscreteSystem import DelayedDiscreteSystem
import theano
import theano.tensor as T

class KerasToDelayedSystemWrapper(DelayedDiscreteSystem):
    def __init__(self,model,params):
        DelayedDiscreteSystem.__init__(self)
        self.dt=params['dt']
        self.xdim=params['xdim']
        self.udim=params['udim']
        self.r=params['xdelay']
        
        self.xu_flat=T.vector()
        self.xnew = theano.clone(model.y_test, replace={model.X_test:self.xu_flat.dimshuffle('x',0)}, share_inputs=True).T
        
        DelayedDiscreteSystem.init_theano_vars_joined(self)