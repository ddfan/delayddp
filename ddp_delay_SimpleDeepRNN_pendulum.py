
### PROFILING
# f = open('profile_log.txt', 'w')
# import cProfile, pstats, StringIO
# pr = cProfile.Profile()
# pr.enable()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DDPDelay import DDPDelay
from Pendulum import Pendulum
import math
import scipy.io as sio
# import theano
# theano.config.optimizer='fast_compile'

print 'hello world'

# load pendulum modeled with SimpleDeepRNN
import nndynamics_helpers as helpers
from SimpleDeepRNNToDelayedSystemWrapper import SimpleDeepRNNToDelayedSystemWrapper
dataprefix='pendulum'
#load model
systeminfo=helpers.load_data("".join([dataprefix,"_model.info"]))
sys=SimpleDeepRNNToDelayedSystemWrapper(systeminfo)
dt=systeminfo['dt']
r=systeminfo['xdelay']-1
xdim=systeminfo['xdim']
udim=systeminfo['udim']
udelay=systeminfo['udelay']
hidden_dim=sys.hidden_dim
sys.xdim=hidden_dim
N=sys.N

# quadratic loss
from QuadraticLossDelayed import QuadraticLossDelayedAnglePendulum
xT=np.zeros((1,1))
xT[0,:]=math.pi
xcost=np.ones((1,1))*1.0
ucost=np.array([[0.0008]])
diffcost=np.ones((1,1))*0.0
lossparams={'dt':dt, 'xdim':hidden_dim, 'udim':sys.udim, 'xcost':xcost, 'ucost':ucost, 'diffcost':diffcost, 'xT':xT, 'r':r+1}
loss=QuadraticLossDelayedAnglePendulum(lossparams)

# setup ddp
Horizon=N
pad_length=systeminfo['pad_length']
x0_model=sys.runSystem(np.zeros((hidden_dim,r+1)),np.zeros((udim,pad_length)))[:,-(r+1):]
ddpparams={'Horizon':Horizon,'gamma':0.2,'min_gamma':0.001,'gamma_factor':3.0,'mu':1.0,'second_order':False,'r':r,'dt':dt,'sys':sys,'loss':loss,'verbose':True}
ddp=DDPDelay(ddpparams)
uk0=np.zeros((sys.udim,Horizon))
print 'running ddp...'
ddp.run(x0_model,uk0)

# pr.disable()
# s = StringIO.StringIO()
# ps = pstats.Stats(pr, stream=s)
# ps.sort_stats('time', 'cumulative').print_stats(.5, 'init')
# ps.print_stats()
# ps.print_callers(.5, 'init')
# print >> f, s.getvalue()

#apply control to real system
x0=np.array([[0.0],[0.0]])
realsys=Pendulum(systeminfo['params'])
print 'running system...'
xtraj_sys=realsys.splitAngle(realsys.runSystem(x0,ddp.uk))
print 'running model...'
xtraj_model=sys.runSystem(x0_model,uk0)
print 'running system with feedback'
xtraj_sys_feedback=realsys.splitAngle(realsys.runSystemWithFeedback(x0,ddp.uk,ddp.K[:,:xdim,0,:],xtraj_model[:xdim,:],feedback_split=True))

print 'plotting...'
fig,axes=plt.subplots(nrows=xdim+1,ncols=1)
pd.DataFrame(ddp.uk.T).plot(ax=axes[0],legend=False,title='u')
for i in xrange(xdim):
    pd.DataFrame(np.vstack([xtraj_model[i,-Horizon:],xtraj_sys[i,:],xtraj_sys_feedback[i,:]]).T,columns=['model','sys','sys w feedback']).plot(ax=axes[i+1],title=i)

sio.savemat('pendulum_model_delayddp.mat',{'u':ddp.uk,'model':xtraj_model[:,-Horizon:],'system':xtraj_sys,'sysfeedback':xtraj_sys_feedback})
                                           
plt.show()