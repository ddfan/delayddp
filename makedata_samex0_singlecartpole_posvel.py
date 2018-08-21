from SingleCartPole import SingleCartPole
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import nndynamics_helpers as helpers

print 'hello world'

# single cart pole parameters
dt=0.01
L=1.0
params={'dt':dt, 'M':0.1, 'm':0.1, 'g':9.8, 'L':L,'k1':0.001,'k2':0.001}
x0=np.array([[0.0],[math.pi],[0.0],[0.0]])
sys=SingleCartPole(params)

M=50000
N=50
uscale=100.0
print 'making data...'
#make input
freq=np.logspace(-0.01,1.8,num=M)
xtraj=sys.runSystem(x0,np.zeros((1,N)))
udata=np.zeros((1,N))
for i in range(0,M/4):
    while True:
        u=np.random.uniform(-uscale,uscale,size=(1,N))
        newtraj=sys.runSystem(x0,u)
        if np.isnan(np.amax(newtraj)) or np.amax(np.abs(newtraj))>20.0:
            continue
        else:
            xtraj=np.dstack([xtraj,newtraj])
            udata=np.dstack([udata,u])
            break
    
for i in range(0,M):
    while True:
        u=np.sin(np.random.uniform(-math.pi,math.pi)+freq[i]*dt*np.linspace(1.0,N,num=N))*np.random.uniform(-uscale,uscale)
        u=np.expand_dims(np.array(u),axis=0)
        newtraj=sys.runSystem(x0,u)
        if np.isnan(np.amax(newtraj)) or np.amax(np.abs(newtraj))>20.0:
            continue
        else:
            xtraj=np.dstack([xtraj,newtraj])
            udata=np.dstack([udata,u])
            break
    while True:
        u=(np.random.uniform(-uscale,uscale)*np.sin(np.random.uniform(-math.pi,math.pi)+freq[i]*dt*np.linspace(1.0,N,num=N))
           -np.random.uniform(uscale/2.0,uscale*1.2)*np.exp(-np.linspace(0.0,N,num=N)/np.random.uniform(1.0,30.0)))
        u=np.expand_dims(np.array(u),axis=0)
        newtraj=sys.runSystem(x0,u)
        if np.isnan(np.amax(newtraj)) or np.amax(np.abs(newtraj))>20.0:
            continue
        else:
            xtraj=np.dstack([xtraj,newtraj])
            udata=np.dstack([udata,u])
            break

xtraj=xtraj.T
udata=udata.T

WRITEDATA=True
ANIMATE=False
PLOTDATA=False

# transform data for nn
xudata=np.zeros((xtraj.shape[0],xtraj.shape[1],5))
xudata[:,:,:4]=xtraj
xudata[:,:,4]=-np.cos(xtraj[:,:,1])
xudata[:,:,1]=np.sin(xtraj[:,:,1])
xudata=np.dstack((xudata,udata))
row_sums = np.zeros((1,1,6))
row_std = np.ones((1,1,6))
row_std[0,0,:-1]=np.array([-np.amin(np.amin(xudata,axis=0),axis=0),np.amin(np.amax(xudata,axis=0),axis=0)]).max(axis=0)[:-1]
print row_std
xudata=(xudata-row_sums) / row_std

x0=np.vstack([x0,np.zeros((1,1))])
x0[4,:]=-np.cos(x0[1,:])
x0[1,:]=np.sin(x0[1,:])
x0=np.expand_dims(x0.T,axis=0)
x0=(x0-row_sums[:,:,-1:])/row_std[:,:,-1:]

#write data
if WRITEDATA:
    print "writing data..."
    systeminfo={'params':params, 'row_sums':row_sums,'row_std':row_std,'dt':dt,'xdim':xtraj.shape[2],'udim':udata.shape[2],'N':N,'x0':x0,'uscale':uscale}
    helpers.save_data([xudata,systeminfo], "singlecartpole_posvel_data.p" )

if PLOTDATA:
    print 'plotting...'
    fig,axes=plt.subplots(nrows=6,ncols=1)
    pd.DataFrame(xudata[:,:,5].T).plot(ax=axes[0],legend=False)
    pd.DataFrame(xudata[:,:,0].T).plot(ax=axes[1],legend=False)
    pd.DataFrame(xudata[:,:,1].T).plot(ax=axes[2],legend=False)
    pd.DataFrame(xudata[:,:,2].T).plot(ax=axes[3],legend=False)
    pd.DataFrame(xudata[:,:,3].T).plot(ax=axes[4],legend=False)
    pd.DataFrame(xudata[:,:,4].T).plot(ax=axes[5],legend=False)
    

def init():
    """initialize animation"""
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def animate(i):
    """perform animation step"""
    global xtraj, dt, L
    
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