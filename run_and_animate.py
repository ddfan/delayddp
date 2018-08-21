
from DoubleCartPole import DoubleCartPole
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

print('hello world')

L1=1.0
L2=1.0
dt=0.01
M=4
n=1000
uscale=1.0
params={'dt':dt, 'm0':0.1, 'm1':0.1, 'm2':0.1, 'L1':L1, 'L2':L2, 'g':9.8, 'k1':0.001, 'k2':0.01, 'k3':0.01}
x0=np.array([[0.0],[math.pi],[math.pi],[0.0],[0.0],[0.0]])

#make input
u=[]
for i in range(0,M):
    u=np.hstack([u,math.pow(-1.0,i)*np.sin(2.0*dt*np.linspace(1.0,n,num=n))*uscale])
#     u=np.hstack([u,math.pow(-1.0,i)*np.array([random.gauss(0,uscale/2.0) for j in range(0,int(n))])])
#     u=np.hstack([u,math.pow(-1.0,i)*np.sin(0.001*dt*np.power(np.linspace(1.0,n,num=n),2))*uscale])
u=np.expand_dims(np.array(u),axis=0)

sys=DoubleCartPole(params)
print('running system...')
xtraj=sys.runSystem(x0,u)

pd.DataFrame(u.T).plot()
pd.DataFrame(xtraj.T).plot()
print('plotting...')
# set up figure and animation
# set up figure and animation
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-5, 5), ylim=(-2, 2))
ax.grid()
 
line, = ax.plot([], [], 'o-', lw=2)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
 
def init():
    """initialize animation"""
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text
 
def animate(i):
    """perform animation step"""
    global xtraj, dt, L1, L2
    
    line.set_data([-0.5+xtraj[0,i],0.5+xtraj[0,i],+xtraj[0,i],math.sin(xtraj[1,i])*L1+xtraj[0,i],math.sin(xtraj[1,i])*L1+math.sin(xtraj[2,i])*L2+xtraj[0,i]],
                   [0,0,0,math.cos(xtraj[1,i])*L1,math.cos(xtraj[1,i])*L1+math.cos(xtraj[2,i])*L2])
    time_text.set_text('time = %.2f' % (dt*i))
    return line, time_text
 
# choose the interval based on dt and the time to run_and_animate one step
from time import time
t0 = time()
animate(0)
t1 = time()
interval = 10000 * dt - (t1 - t0)

ani = animation.FuncAnimation(fig, animate, frames=n*M,
                              interval=interval, blit=True, init_func=init)
# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
#ani.save('double_pendulum.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()