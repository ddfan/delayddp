from DelayedDiscreteSystem import DelayedDiscreteSystem
import theano
import theano.tensor as T

class DeepRNNToDelayedSystemWrapper(DelayedDiscreteSystem):
    def __init__(self,params):
        DelayedDiscreteSystem.__init__(self)
        self.dt=params['dt']
        self.xdim=params['xdim']
        self.udim=params['udim']
        self.r=params['xdelay']
        self.U1=params['U1']
        self.U2=params['U2']
        self.U3=params['U3']
        self.b1=params['b1']
        self.b2=params['b2']
        self.hidden_dim=params['output_dim']
        self.inner_activation=params['inner_activation']
        self.activation=params['activation']
        self.N=params['N']
        
        xvec=self.xu_flat[-self.udim:]
        for i in range(self.r):
            xvec=T.concatenate([xvec,self.xu_flat[(self.r-i-1)*self.xdim:(self.r-i)*self.xdim]])
        output=T.dot(self.activation( T.dot(self.inner_activation(T.dot(xvec,self.U1[0]) + self.b1[0]),self.U2[0]) + self.b2[0] ),self.U3[0])
        for i in range(1,self.hidden_dim):
            output=T.concatenate([output,T.dot(self.activation( T.dot(self.inner_activation(T.dot(xvec,self.U1[i]) + self.b1[i]),self.U2[i]) + self.b2[i] ),self.U3[i])])
        self.xnew=output.dimshuffle(0,'x')
        
        DelayedDiscreteSystem.init_theano_vars(self)