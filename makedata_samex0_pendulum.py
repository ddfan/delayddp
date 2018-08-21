from Pendulum import Pendulum
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import nndynamics_helpers as helpers

print 'hello world'

# pendulum parameters
dt=0.02
L=1.0
params={'dt':dt, 'c':0.01, 'm':0.1, 'g':9.8, 'l':L}
x0=np.array([[0.0],[0.0]])
sys=Pendulum(params)

M=10000
N=50
uscale=100.0
print 'making data...'
#make input
freq=np.logspace(-0.01,1.8,num=M)
xtraj=sys.runSystem(x0,np.zeros((1,N)))
udata=np.zeros((1,N))
for i in range(0,M/4):
    u=np.random.uniform(-uscale,uscale,size=(1,N))
    xtraj=np.dstack([xtraj,sys.runSystem(x0,u)])
    udata=np.dstack([udata,u])
for i in range(0,M):
    u=np.sin(np.random.uniform(-math.pi,math.pi)+freq[i]*dt*np.linspace(1.0,N,num=N))*np.random.uniform(-uscale,uscale)
    u=np.expand_dims(np.array(u),axis=0)
    xtraj=np.dstack([xtraj,sys.runSystem(x0,u)])
    udata=np.dstack([udata,u])

xtraj=xtraj.T
udata=udata.T

WRITEDATA=True
ANIMATE=False
PLOTDATA=False

# transform data for nn
xtraj[:,:,1]=-np.cos(xtraj[:,:,0])
xtraj[:,:,0]=np.sin(xtraj[:,:,0])
xudata=np.dstack((xtraj,udata))
row_sums = np.array([[[0,0,0]]])
row_std = np.array([[[1,1,1]]])
xudata=(xudata-row_sums) / row_std

x0[1,:]=-np.cos(x0[0,:])
x0[0,:]=np.sin(x0[0,:])
x0=np.expand_dims(x0.T,axis=0)
x0=(x0-row_sums[:,:,:1])/row_std[:,:,:1]

#write data
if WRITEDATA:
    print "writing data..."
    systeminfo={'params':params, 'row_sums':row_sums,'row_std':row_std,'dt':dt,'xdim':xtraj.shape[2],'udim':udata.shape[2],'N':N,'x0':x0,'uscale':uscale}
    helpers.save_data([xudata,systeminfo], "pendulum_data.p" )

if PLOTDATA:
    print 'plotting...'
    fig,axes=plt.subplots(nrows=3,ncols=1)
    pd.DataFrame(udata[:,:,0].T).plot(ax=axes[0],legend=False)
    pd.DataFrame(xtraj[:,:,0].T).plot(ax=axes[1],legend=False)
    pd.DataFrame(xtraj[:,:,1].T).plot(ax=axes[2],legend=False)

def init():
    """initialize animation"""
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def animate(i):
    """perform animation step"""
    global xtraj, dt, L1, L2
    
    line.set_data([(xtraj[0,i,0])*L,0],[(xtraj[0,i,1])*L,0])
    time_text.set_text('time = %.2f' % (dt*i))
    return line, time_text
 
if ANIMATE:
    # set up figure and animation
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-5, 5), ylim=(-2, 2))
    ax.grid()
 
    line, = ax.plot([], [], 'o-', lw=2)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
 
    # choose the interval based on dt and the time to run_and_animate one step
    from time import time
    t0 = time()
    animate(0)
    t1 = time()
    interval = 1000 * dt - (t1 - t0)
    ani = animation.FuncAnimation(fig, animate, frames=N*M,
                              interval=interval, blit=True, init_func=init)
    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    #ani.save('double_pendulum.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
if PLOTDATA:
    plt.show()