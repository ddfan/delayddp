# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 15:58:53 2015
@author: david
"""

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import random
np.random.seed(1337) # for reproducibility
random.seed(1337)

import nndynamics_helpers as helpers
from Pendulum import Pendulum 
from KerasToDelayedSystemWrapper import KerasToDelayedSystemWrapper

dataprefix='pendulum'

#load model
model,systeminfo=helpers.load_model(dataprefix)
modelsys=KerasToDelayedSystemWrapper(model,systeminfo)

dt=systeminfo['dt']
xdelay=systeminfo['xdelay']
xdim=systeminfo['xdim']
udim=systeminfo['udim']
udelay=systeminfo['udelay']

#load system
x0=np.array([[math.pi],[0.0]])
sys=Pendulum(systeminfo['params'])

#make input
print 'making data...'
M=3
n=50
uscale=5.0
u=[]
for i in range(0,M):
#     u=np.hstack([u,math.pow(-1.0,i)*np.sin(2.0*dt*np.linspace(1.0,n,num=n))*uscale])
    u=np.hstack([u,-math.pow(-1.0,i)*np.sin(0.01*dt*np.power(np.linspace(1.0,n,num=n),2))*uscale])
    u=np.hstack([u,math.pow(-1.0,i)*np.array([random.uniform(-uscale,uscale) for j in range(0,int(n))])])
u=np.expand_dims(np.array(u),axis=0)

print 'running system...'
xtraj_sys=sys.runSystem(x0,u)
xtraj_sys[1,:]=-np.cos(xtraj_sys[0,:])
xtraj_sys[0,:]=np.sin(xtraj_sys[0,:])
xudata=np.vstack([xtraj_sys,u])
row_sums=systeminfo['row_sums']
row_std=systeminfo['row_std']
xudata=xudata-np.array([row_sums]).T
xudata=xudata/np.array([row_std]).T

print 'running model...'
xtraj_model=xudata[0:xdim,0:xdelay]
for i in xrange(u.shape[1]-xdelay):
    xtraj_model = np.hstack([xtraj_model,modelsys.getNextState(xtraj_model[:,-xdelay:],xudata[-udim:,i-udelay])])

#plot
for i in xrange(xdim):
    pd.DataFrame(np.vstack([xtraj_sys[i,:],xtraj_model[i,:]]).T).plot()
plt.show()