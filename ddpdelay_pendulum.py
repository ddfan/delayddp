
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DDPDelay import DDPDelay
import nndynamics_helpers as helpers
from KerasToDelayedSystemWrapper import KerasToDelayedSystemWrapper

# load keras model
dataprefix='pendulum'
model,systeminfo=helpers.load_model(dataprefix)
sys=KerasToDelayedSystemWrapper(model,systeminfo)


# quadratic loss
from QuadraticLossDelayed import QuadraticLossDelayed
xT=np.array([[0.0],[1.0]])
xcost=np.array([[1.0], [1.0]])
ucost=np.array([[0.01]])
lossparams={'dt':sys.dt, 'xdim':sys.xdim, 'udim':sys.udim, 'xcost':xcost, 'ucost':ucost, 'xT':xT, 'r':sys.r}
loss=QuadraticLossDelayed(lossparams)

# setup ddp
Horizon=50
num_iter=100
x0=np.repeat(np.array([[0.0],[-1.0]]),sys.r,axis=1)
ddpparams={'Horizon':Horizon,'num_iter':num_iter,'gamma':0.01,'mu':20.0,'n_switch':num_iter,'r':sys.r-1,'dt':sys.dt,'sys':sys,'loss':loss,'verbose':True}
ddp=DDPDelay(ddpparams)
uk0=np.zeros((sys.udim,Horizon))
print('running ddp...')
ddp.run(x0,uk0)

# pr.disable()
# s = StringIO.StringIO()
# ps = pstats.Stats(pr, stream=s)
# ps.sort_stats('time', 'cumulative').print_stats(.5, 'init')
# ps.print_stats()
# ps.print_callers(.5, 'init')
# print >> f, s.getvalue()

pd.DataFrame(ddp.uk.T).plot()
pd.DataFrame(ddp.xtraj.T).plot()

print('plotting...')

plt.show()