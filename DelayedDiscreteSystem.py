# class for a discrete time system.  includes methods for getting gradients and so forth.  requires theano.

import theano
import theano.tensor as T
import numpy as np
import numpy.matlib

class DelayedDiscreteSystem:
    def __init__(self):
        self.dt=1
        self.xdim=1
        self.udim=1
        self.r=1
        self.delay=T.bscalar()
        self.delay2=T.bscalar()
        self.x=T.matrix()
        self.u=T.col()
        self.xu_flat=T.concatenate([T.flatten(self.x.T),T.flatten(self.u)])
        
    def init_theano_vars(self):
        self.f=theano.function([self.x,self.u],self.xnew,allow_input_downcast=True)
        
        J=theano.gradient.jacobian(T.flatten(self.xnew),self.xu_flat)
        self.fx=theano.function([self.x,self.u],J[:,0:-self.udim],allow_input_downcast=True)
        self.fu=theano.function([self.x,self.u],J[:,-self.udim:],allow_input_downcast=True)
        
        H, updates=theano.scan(lambda i, xnew,xu: theano.gradient.hessian(xnew[i],xu), sequences=T.arange(T.flatten(self.xnew).shape[0]), non_sequences=[T.flatten(self.xnew),self.xu_flat])
        self.fxx=theano.function([self.x,self.u],H[:,0:-self.udim,0:-self.udim], updates=updates,allow_input_downcast=True)
        self.fxu=theano.function([self.x,self.u],H[:,-self.udim:,0:-self.udim], updates=updates,allow_input_downcast=True)
        self.fuu=theano.function([self.x,self.u],H[:,-self.udim:,-self.udim:], updates=updates,allow_input_downcast=True)
    
        self.joinedvars=False
        
    def init_theano_vars_joined(self):
        self.f=theano.function([self.xu_flat],self.xnew,allow_input_downcast=True)
        
        J=theano.gradient.jacobian(T.flatten(self.xnew),self.xu_flat)
        self.fx=theano.function([self.xu_flat],J[:,0:-self.udim],allow_input_downcast=True)
        self.fu=theano.function([self.xu_flat],J[:,-self.udim:],allow_input_downcast=True)
        
        H, updates=theano.scan(lambda i, xnew,xu: theano.gradient.hessian(xnew[i],xu), sequences=T.arange(T.flatten(self.xnew).shape[0]), non_sequences=[T.flatten(self.xnew),self.xu_flat])
        self.fxx=theano.function([self.xu_flat],H[:,0:-self.udim,0:-self.udim], updates=updates,allow_input_downcast=True)
        self.fxu=theano.function([self.xu_flat],H[:,-self.udim:,0:-self.udim], updates=updates,allow_input_downcast=True)
        self.fuu=theano.function([self.xu_flat],H[:,-self.udim:,-self.udim:], updates=updates,allow_input_downcast=True)
    
        self.joinedvars=True
        
    def getNextState(self,x_in,u_in):
        if self.joinedvars:
            return self.f(np.hstack([x_in.flatten(),u_in.flatten()]))
        else:
            return self.f(x_in,u_in)
    
    def runSystem(self,x0,u):
        x=x0
        for i in range(0,u.shape[1]):
            x=np.append(x,self.getNextState(x[:,-(self.r):],u[:,i:i+1]),axis=1)
        return x
    
    def runSystemWithSequence(self,xseq,useq):
        valout=numpy.matlib.repmat(np.zeros_like(self.getNextState(xseq[:,0:self.r],useq[:,0:1])),1,xseq.shape[1]-self.r)
        for i in range(0,xseq.shape[1]-self.r):
            valout[:,i:i+1]=self.getNextState(xseq[:,i:i+self.r],useq[:,i:i+1])
        return valout

    def getFx(self,x_in,u_in):
        #get theano gradient of f at x_in, u_in
        if self.joinedvars:
            J=self.fx(np.hstack([x_in.flatten(),u_in.flatten()]))
            fx=np.zeros((J.shape[0],self.xdim,self.r))
            for j in range(self.r):
                fx[:,:,j]=J[:,(self.r-1-j)::self.r]
        else:
            J=self.fx(x_in,u_in)
            fx=np.zeros((J.shape[0],self.xdim,self.r))
            for j in range(self.r):
                fx[:,:,j]=J[:,(self.r-j-1)*self.xdim:(self.r-j)*self.xdim]
        return fx
    
    def getFu(self,x_in,u_in):
        #get theano gradient of f at x_in, u_in
        if self.joinedvars:
            return self.fu(np.hstack([x_in.flatten(),u_in.flatten()]))
        else:
            return self.fu(x_in,u_in)
    
    def getFxx(self,x_in,u_in):
        if self.joinedvars:
            H=self.fxx(np.hstack([x_in.flatten(),u_in.flatten()]))
            fxx=np.zeros((self.xdim,self.xdim,H.shape[0],self.r,self.r))
            for j in range(self.r):
                for l in range(self.r):
                    fxx[:,:,:,j,l]=H[:,(self.r-1-j)::self.r,(self.r-1-l)::self.r].T
        else:
            H=self.fxx(x_in,u_in)
            fxx=np.zeros((self.xdim,self.xdim,H.shape[0],self.r,self.r))
            for j in range(self.r):
                for l in range(self.r):
                    fxx[:,:,:,j,l]=H[:,(self.r-j-1)*self.xdim:(self.r-j)*self.xdim,(self.r-l-1)*self.xdim:(self.r-l)*self.xdim].T
        return fxx
    
    def getFxu(self,x_in,u_in):
        if self.joinedvars:
            H=self.fxu(np.hstack([x_in.flatten(),u_in.flatten()]))
            fxu=np.zeros((self.xdim,self.udim,H.shape[0],self.r))
            for j in range(self.r):
                fxu[:,:,:,j]=H[:,-self.udim:,(self.r-1-j)::self.r].T
        else:
            H=self.fxu(x_in,u_in)
            fxu=np.zeros((self.xdim,self.udim,H.shape[0],self.r))
            for j in range(self.r):
                fxu[:,:,:,j]=H[:,-self.udim:,(self.r-j-1)*self.xdim:(self.r-j)*self.xdim].T
        return fxu
    
    def getFuu(self,x_in,u_in):
        if self.joinedvars:
            return self.fuu(np.hstack([x_in.flatten(),u_in.flatten()])).T
        else:
            return self.fuu(x_in,u_in).T
    

    