# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 15:58:53 2015
@author: david
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
np.random.seed(1337) # for reproducibility
random.seed(1337)

import keras.models
from keras.layers.core import Lambda, TimeDistributedDense
from keras.layers.recurrent import SimpleDeepRNN
from keras.optimizers import SGD
import nndynamics_helpers as helpers
from keras.layers.noise import GaussianNoise, GaussianDropout
from keras.optimizers import Adam
# import theano
#theano.config.optimizer='fast_run'
#theano.config.exception_verbosity='low'

xdelay = 4
SAVEMODEL=True;
dataprefix='singlecartpole_posvel'
udelay=1
batch_size=128
epochs=15
hidden_size=200

#load data
print "prepping data..."
xudata,systeminfo=helpers.load_data("".join([dataprefix,"_data.p"]))
xdim=5
udim=systeminfo['udim']
pad_length=xdelay
N=systeminfo['N']+pad_length
x0=systeminfo['x0']
num_trials=xudata.shape[0]
indim=udim
outdim=xdim
x=np.hstack([np.zeros((1,1,udim)).repeat(pad_length+udelay,1).repeat(xudata.shape[0],0),xudata[:,:-udelay,-udim:]])
y=np.hstack([x0.repeat(pad_length,1).repeat(xudata.shape[0],0),xudata[:,:,:xdim]])
X_test,y_test,X_train,y_train = helpers.train_test_split(x,y)

def splitfcn(input_tensor):
    xdim=5
    return input_tensor[:,:,0:xdim]

#build model
print "building model..."  
model = keras.models.Sequential()
model.add(GaussianNoise(0.1,input_shape=(N,indim,)))
# model.add(GaussianDropout(0.02,input_shape=(N,indim,)))
model.add(SimpleDeepRNN(hidden_size, input_shape=(N,indim,), depth=xdelay, activation='tanh', init='glorot_normal', truncate_gradient=systeminfo['N'], inner_activation='tanh', inner_init='glorot_normal', return_sequences=True))
model.add(Lambda(splitfcn,output_shape=(N,xdim,)))
# model.add(TimeDistributedDense(outdim))
adamopt=Adam(clipnorm=0.1)
model.compile(loss="mse", optimizer=adamopt)

# earlystop=keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0)
# 
# # train the model
print "training..."
# model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs, validation_split=0.05, verbose=2, show_accuracy=True, callbacks=[earlystop])
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs, validation_split=0.05, verbose=2)

predicted = model.predict(X_test)
rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))
print np.mean(rmse,axis=0)

#compare predicted and y_test
plotnum=3
for j in xrange(plotnum):
    fig,axes=plt.subplots(nrows=xdim+1,ncols=1)
    pd.DataFrame(X_test[j,:,0].T).plot(ax=axes[0])
    for i in xrange(xdim):
        pd.DataFrame(np.vstack([predicted[j,:,i],y_test[j,:,i]]).T).plot(ax=axes[i+1])
    
if SAVEMODEL:
    print "saving model..."
    systeminfo['xdelay']=xdelay
    systeminfo['udelay']=udelay
    systeminfo['indim']=indim
    systeminfo['outdim']=outdim
    systeminfo['pad_length']=pad_length
    
    #save model weights manually
    systeminfo['W']=model.layers[1].W.get_value()
    systeminfo['b']=model.layers[1].b.get_value()
    systeminfo['output_dim']=model.layers[1].output_dim
    Us=[]
    for i in xrange(len(model.layers[1].Us)):
        Us.append(model.layers[1].Us[i].get_value())
    systeminfo['Us']=Us
    systeminfo['inner_activation']=model.layers[1].inner_activation
    systeminfo['activation']=model.layers[1].activation

    helpers.save_data(systeminfo,"".join([dataprefix,"2_model.info"]))
        
plt.show()