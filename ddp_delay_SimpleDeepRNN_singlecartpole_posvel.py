
### PROFILING
# f = open('profile_log.txt', 'w')
# import cProfile, pstats, StringIO
# pr = cProfile.Profile()
# pr.enable()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DDPDelay import DDPDelay
from SingleCartPole import SingleCartPole
import math
import scipy.io as sio
import random
# import theano
# theano.config.optimizer='fast_compile'
np.random.seed(1337) # for reproducibility
random.seed(1337)

print 'hello world'

# load pendulum modeled with SimpleDeepRNN
import nndynamics_helpers as helpers
from SimpleDeepRNNToDelayedSystemWrapper import SimpleDeepRNNToDelayedSystemWrapper
dataprefix='singlecartpole_posvel2'
#load model
systeminfo=helpers.load_data("".join([dataprefix,"_model.info"]))
sys=SimpleDeepRNNToDelayedSystemWrapper(systeminfo)
dt=systeminfo['dt']
r=systeminfo['xdelay']-1
xdim=systeminfo['xdim']+1
udim=systeminfo['udim']
udelay=systeminfo['udelay']
hidden_dim=sys.hidden_dim
sys.xdim=hidden_dim
# N=sys.N
N=50
uscale=1.0
row_std=systeminfo['row_std'][0,0,:]

# quadratic loss
from QuadraticLossDelayed import QuadraticLossDelayedAngleSingleCartPolePosVel
xT=np.zeros((4,1))
# xcost=np.array([[100.0], [10.0], [0.01], [20.0]])
# ucost=np.array([[0.05]])
xcost=np.array([[5.0*row_std[0]], [40.0], [0.01*row_std[2]], [1.0*row_std[3]]])
ucost=np.array([[0.1]])
lossparams={'dt':dt, 'xdim':hidden_dim, 'udim':sys.udim, 'xcost':xcost, 'ucost':ucost, 'xT':xT, 'r':r+1}
loss=QuadraticLossDelayedAngleSingleCartPolePosVel(lossparams)
# xcost=np.array([[10.0*row_std[0]], [200.0], [200.0], [0.0000001*row_std[3]], [0.001*row_std[4]], [0.001*row_std[5]]])
# ucost=np.array([[0.2]])
# setup ddp
Horizon=N
pad_length=systeminfo['pad_length']
x0_model=sys.runSystem(np.zeros((hidden_dim,r+1)),np.zeros((udim,pad_length)))[:,-(r+1):]
ddpparams={'Horizon':Horizon,'num_iter':9999,'schedule':[0.5,0.5,0.5,0.5,0.5,0.5,0.1],'discount':1.0,'stopratio':0.1,'gamma':0.3,'min_gamma':0.001,'gamma_factor':2.0,'mu':1.0,'second_order':False,'r':r,'dt':dt,'sys':sys,'loss':loss,'verbose':True}
ddp=DDPDelay(ddpparams)
# [0.01,0.8,0.8,0.8,0.8,0.8,0.3,0.3,0.3,0.3,0.1]
uk0=uscale*np.random.randn(sys.udim,Horizon)
# uk0=np.array([[  2.21708357e-01,  -6.39219544e+01,  -5.24766873e+01,  -4.65513679e+01,
#    -3.03113670e+01,  -1.68965443e+01,  -8.28860106e+00,  -1.19244462e+01,
#     2.50926895e+00,   3.70815770e+00,   3.79576168e+00,   4.20896916e+00,
#     6.78590879e+00,  -2.93728365e+00,  -8.51609915e+00,  -9.29269702e+00,
#    -1.06148212e+01,  -1.54417414e+01,  -1.75656069e+01,  -2.04741577e+01,
#    -2.01818808e+01,  -2.18178107e+01,  -2.01684072e+01,  -1.86428920e+01,
#    -1.82400460e+01,  -1.58286074e+01,  -1.30064916e+01,  -1.11093395e+01,
#    -1.04289587e+01,  -9.15429170e+00,  -7.76614311e+00,  -6.74888425e+00,
#    -5.86909635e+00,  -5.15995761e+00,  -4.58843673e+00,  -4.34249242e+00,
#    -4.01562587e+00,  -3.61768262e+00,  -3.42282679e+00,  -2.97138014e+00,
#    -2.34962404e+00,  -2.12917000e+00,  -1.96702329e+00,  -1.60449124e+00,
#    -1.43155575e+00,  -1.43093224e+00,  -4.89567640e-01,  -1.55809603e-01,
#     7.60843940e-02,  -7.70721396e-03]])
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
x0=np.array([[0.0],[math.pi],[0.0],[0.0]])
systeminfo['params']['row_sums']=systeminfo['row_sums']
systeminfo['params']['row_std']=systeminfo['row_std']
realsys=SingleCartPole(systeminfo['params'])
print 'running system...'
xtraj_sys=realsys.splitAngle(realsys.runSystem(x0,ddp.uk),posvel=True)
print 'running model...'
xtraj_model=sys.runSystem(x0_model,uk0)
print 'running system with feedback'
xtraj_sys_feedback=realsys.splitAngle(realsys.runSystemWithFeedback(x0,ddp.uk,ddp.K[:,:xdim,0,:],xtraj_model[:xdim,:],feedback_split=True, posvel=True),posvel=True)

print 'plotting...'
fig,axes=plt.subplots(nrows=xdim+1,ncols=1)
pd.DataFrame(ddp.uk.T).plot(ax=axes[0],legend=False,title='u')
for i in xrange(xdim):
#     pd.DataFrame(np.vstack([xtraj_model[i,-Horizon:],xtraj_sys[i,:],xtraj_sys_feedback[i,:]]).T,columns=['model','sys','sys w feedback']).plot(ax=axes[i+1],title=i)
    pd.DataFrame(np.vstack([xtraj_model[i,-Horizon:],xtraj_sys[i,:],xtraj_sys_feedback[i,:]]).T).plot(legend=False,ax=axes[i+1],title=i)

sio.savemat("".join([dataprefix,".mat"]),{'u':ddp.uk,'model':xtraj_model[:,-Horizon:],'system':xtraj_sys,'sysfeedback':xtraj_sys_feedback})

print ddp.uk

# print 'plotting...'
# set up figure and animation
import matplotlib.animation as animation
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-15, 1), ylim=(-2, 2))
ax.grid()
   
line1, = ax.plot([], [], 'o-', lw=2, c='b')
line2, = ax.plot([], [], 'o-', lw=2, c='r')
time_text = ax.text(0.1, 0.9, '', transform=ax.transAxes)
   
def init():
    """initialize animation"""
    line1.set_data([], [])
    line2.set_data([], [])
    time_text.set_text('')
    return line1,line2, time_text
   
def animate(i):
    """perform animation step"""
    global xtraj_sys_feedback, dt
    L=1.0
    xtraj=xtraj_sys_feedback
    line1.set_data([-0.5+xtraj[0,i]*10,0.5+xtraj[0,i]*10,xtraj[0,i]*10,xtraj[1,i]*L+xtraj[0,i]*10],
                    [0,0,0,-xtraj[4,i]*L])
    xtraj=xtraj_model
    line2.set_data([-0.5+xtraj[0,i]*10,0.5+xtraj[0,i]*10,+xtraj[0,i]*10,xtraj[1,i]*L+xtraj[0,i]*10],
                    [0,0,0,-xtraj[4,i]*L])
    time_text.set_text('time = %.2f' % (dt*i))
    return line1, line2, time_text
   
# choose the interval based on dt and the time to run_and_animate one step
from time import time
t0 = time()
animate(0)
t1 = time()
interval = 10000 * dt - (t1 - t0)
  
ani = animation.FuncAnimation(fig, animate, frames=Horizon,
                              interval=interval, blit=True, init_func=init)
  
# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
# ani.save("".join([dataprefix,".mp4"]), fps=30, extra_args=['-vcodec', 'libx264'])
plt.show()


# 
# 
# hello world
# running ddp...
# Iteration #: 0 Cost: 2334.85489854
# 0.5
# Iteration #: 1 Cost: 1060.38090776
# 0.5
# Iteration #: 2 Cost: 968.646937999
# 0.5
# Iteration #: 3 Cost: 907.886842541
# 0.1
# Iteration #: 4 Cost: 897.945233135
# 0.1
# Iteration #: 5 Cost: 881.723405305
# 0.1
# Iteration #: 6 Cost: 872.268422094
# 0.1
# Iteration #: 7 Cost: 866.794985597
# 0.1
# Iteration #: 8 Cost: 861.029438134
# 0.1
# Iteration #: 9 Cost: 856.735619469
# 0.1
# Iteration #: 10 Cost: 851.781130105
# 0.1
# Iteration #: 11 Cost: 846.887047342
# 0.1
# Iteration #: 12 Cost: 844.747184432
# 0.1
# Iteration #: 13 Cost: 842.856290033
# 0.1
# Iteration #: 14 Cost: 841.252638241
# 0.1
# Iteration #: 15 Cost: 839.90094804
# 0.1
# Iteration #: 16 Cost: 838.978565753
# 0.1
# Iteration #: 17 Cost: 838.287197557
# 0.1
# Iteration #: 18 Cost: 837.503901251
# 0.1
# Iteration #: 19 Cost: 836.494508502
# 0.1
# Iteration #: 20 Cost: 835.523427119
# 0.1
# Iteration #: 21 Cost: 834.829562505
# 0.1
# Iteration #: 22 Cost: 834.246055155
# 0.1
# Iteration #: 23 Cost: 833.532183607
# 0.1
# Iteration #: 24 Cost: 832.858786154
# 0.1
# Iteration #: 25 Cost: 832.200213775
# 0.1
# Iteration #: 26 Cost: 831.521303932
# 0.1
# Iteration #: 27 Cost: 831.112608582
# 0.1
# Iteration #: 28 Cost: 830.810642678
# 0.1
# Iteration #: 29 Cost: 830.531218084
# 0.1
# Iteration #: 30 Cost: 830.196855272
# 0.1
# Iteration #: 31 Cost: 829.697647529
# 0.1
# Iteration #: 32 Cost: 828.83452831
# 0.1
# Iteration #: 33 Cost: 827.530899797
# 0.1
# Iteration #: 34 Cost: 826.51530832
# 0.1
# Iteration #: 35 Cost: 826.082497903
# 0.1
# Iteration #: 36 Cost: 826.077698884
# running system...
# running model...
# running system with feedback
# plotting...
# [[ -0.05645966 -34.28052263 -18.0685088  -19.25005248 -17.9952093
#   -15.44593634  -8.00574517  -8.4354658   -1.13663517  -1.232158
#    -3.42884307   3.75606845   0.98220179   4.50292244   7.85957324
#     7.7940008    9.70026239   4.53947262   4.97983328  11.1071436
#     4.39387612   2.86325506   5.99685833   5.3474459    6.53729118
#     5.9542922    4.43274918   1.88463877  -1.79730503   0.67576937
#     2.5127284    3.97881795  -3.46473425  -3.36492587  -4.61382087
#    -1.77867215  -3.72314179  -4.40938074  -4.95344742  -5.51220872
#    -3.90340434  -3.3133456   -2.94749372  -2.48970254  -1.92773329
#    -1.44568039  -0.89820627  -0.50737411  -0.27022574   0.18494977]]