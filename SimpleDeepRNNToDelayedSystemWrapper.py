from DelayedDiscreteSystem import DelayedDiscreteSystem
import theano
import theano.tensor as T

class SimpleDeepRNNToDelayedSystemWrapper(DelayedDiscreteSystem):
    def __init__(self,params):
        DelayedDiscreteSystem.__init__(self)
        self.dt=params['dt']
        self.xdim=params['xdim']
        self.udim=params['udim']
        self.r=params['xdelay']
        self.W=params['W']
        self.b=params['b']
        self.hidden_dim=params['output_dim']
        self.Us=params['Us']
        self.inner_activation=params['inner_activation']
        self.activation=params['activation']
        self.N=params['N']
        
        xnew=T.zeros_like(self.b,dtype='floatX')
        for i in xrange(self.r):
            xnew+=self.inner_activation(T.dot(self.xu_flat[(self.r-i-1)*(self.hidden_dim):(self.r-i)*(self.hidden_dim)],self.Us[i]))
        
        self.xnew = self.activation(T.dot(self.xu_flat[-self.udim:].T,self.W)+self.b + xnew).dimshuffle(0,'x')

        DelayedDiscreteSystem.init_theano_vars(self)