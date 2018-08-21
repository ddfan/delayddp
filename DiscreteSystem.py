# class for a discrete time system.  includes methods for getting gradients and so forth.  requires theano.

import theano
import theano.tensor as T
import copy
import numpy as np
import numpy.matlib

class DiscreteSystem:
    def __init__(self):
        self.dt=1
        self.xdim=1
        self.udim=1
        self.x=T.fcol()
        self.u=T.fcol()
        self.xu_flat=T.flatten(T.concatenate([self.x,self.u]))
        self.xu=self.xu_flat.dimshuffle(0,'x')

    def init_theano_vars(self):
        self.f=theano.function([self.x,self.u],self.xnew,allow_input_downcast=True)
        self.fx=theano.function([self.x,self.u],theano.gradient.jacobian(T.flatten(self.xnew),self.x),allow_input_downcast=True)
        self.fu=theano.function([self.x,self.u],theano.gradient.jacobian(T.flatten(self.xnew),self.u),allow_input_downcast=True)
        
        H, updates=theano.scan(lambda i, xnew,xu: theano.gradient.hessian(xnew[i],xu), sequences=T.arange(T.flatten(self.xnew).shape[0]), non_sequences=[T.flatten(self.xnew),self.xu_flat])
        self.fxx=theano.function([self.x,self.u],H[:,:self.xdim,:self.xdim], updates=updates,allow_input_downcast=True)
        self.fxu=theano.function([self.x,self.u],H[:,:self.xdim,-self.udim:], updates=updates,allow_input_downcast=True)
        self.fuu=theano.function([self.x,self.u],H[:,-self.udim:,-self.udim:], updates=updates,allow_input_downcast=True)
    
    def getNextState(self,x_in,u_in):
        return self.f(x_in,u_in)
    
    def runSystem(self,x0,u):
        x=x0
        for i in range(0,u.shape[1]-1):
            x=np.append(x,self.getNextState(copy.copy(x[:,-1:]),u[:,i:i+1]),axis=1)
        return x
    
    def runSystemWithFeedback(self,x0,u,K,xtrue,feedback_split=False,posvel=False):
        x=x0
        for i in range(0,u.shape[1]-1):
            xfeedback=x[:,-1:]
            if feedback_split:
                xfeedback=self.splitAngle(x[:,-1:],posvel=posvel)
                                
            x=np.append(x,self.getNextState(copy.copy(x[:,-1:]),u[:,i:i+1]+np.dot(K[:,:,i],xfeedback-xtrue[:,i:i+1])),axis=1)
        return x
    
    def runSystemWithSequence(self,xseq,useq):
        valout=numpy.matlib.repmat(np.zeros_like(self.getNextState(xseq[:,0:1],useq[:,0:1])),1,xseq.shape[1])
        for i in range(0,xseq.shape[1]):
            valout[:,i:i+1]=self.getNextState(xseq[:,i:i+1],useq[:,i:i+1])
        return valout

    def getFx(self,x_in,u_in):
        #get theano gradient of f at x_in, u_in
        return np.squeeze(self.fx(x_in,u_in),axis=2)
    
    def getFu(self,x_in,u_in):
        #get theano gradient of f at x_in, u_in
        return np.squeeze(self.fu(x_in,u_in),axis=2)
    
    def getFxx(self,x_in,u_in):
        return self.fxx(x_in,u_in)
    
    def getFxu(self,x_in,u_in):
        return self.fxu(x_in,u_in)
    
    def getFuu(self,x_in,u_in):
        return self.fuu(x_in,u_in)
    

    