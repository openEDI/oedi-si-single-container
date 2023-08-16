"""'
Created on August 15 00:00:00 2022
@author: Argonne National Laboratory
"""

import time
import random
import copy
import pickle
import os
import sys
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cvx

baseDir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"Adding base directory:{baseDir} to path")
sys.path.insert(0,baseDir) #Add module path to prevent import errors

from app.solardisaggregation.disaggregation_utilities import get_data_matrix,get_observed_data,get_aggregate_data

print(f"Base directory:{baseDir}")
modelDir=os.path.join(baseDir,'app','solardisaggregation','model') # modify subdir of solardisaggregation as needed
dataDir=os.path.join(baseDir,'data','solardisaggregation') # modify subdir of solardisaggregation as needed
plotsDir = os.path.join(baseDir,'app','solardisaggregation','plots') # modify subdir of solardisaggregation as needed

if not os.path.isdir(plotsDir):
    print(f"Creating {plotsDir} since it doesn't exist...")
    os.makedirs(plotsDir)

print("Reading lasso regression weights from:lasso_regression_weights.pickle")
with open(os.path.join(modelDir,'lasso_regression_weights.pickle'), 'rb') as handle:
    lasso_regression_weights = pickle.load(handle)

x1 = lasso_regression_weights["x1"]
x2 = lasso_regression_weights["x2"]

datafile = os.path.join(dataDir,"ausgrid.csv")
Data_matrix_o,Data_matrix2,id_num,timelist = get_data_matrix(datafile)

id_num_ob=id_num[1:2+29]
observed_agg,observed_res,observed_solar = get_observed_data(Data_matrix2,timelist,id_num_ob)

observed_res2=list(observed_res.values())
observed_solar2=list(observed_solar.values())
ob_res= np.transpose(np.array(observed_res2))
ob_solar = np.transpose(np.array(observed_solar2))

bb,aggregate_t,Res_gt,Solar_gt,df1,df2 = get_aggregate_data(Data_matrix2,id_num)

selectday2='07/20/2012'
b = np.array( [aggregate_t[selectday2]])
Res_gt_day=np.array([Res_gt[selectday2]])
Solar_gt_day=[np.transpose(Solar_gt[selectday2])]
bb=np.transpose(b)
Res_gt_days = np.transpose(Res_gt_day)
Solar_gt_days = np.transpose(Solar_gt_day)
n=1

print('finished data preparation')

A1 = ob_res.copy()
A2 = ob_solar.copy()
csfont = {'fontname':'Times New Roman'}

tempp=np.transpose(A1@x1.value)
t1 = np.arange(0.0, 48.0, 2)
t1=range(48)
plt.figure(figsize=(8,6))
plt.subplot(311)
plt.plot(t1,bb,linewidth=2.0,label='Net Load',color='b',alpha=0.6)
plt.legend()
#plt.xlabel('Time',fontsize=14,**csfont)
plt.ylabel('Active Power (MW)',fontsize=12,**csfont)
#plt.title('Net Load',fontsize=14, **csfont)
plt.xticks([0, 8-1, 16-1,24-1,32-1,40-1], ['0:00', '4:00', '8:00','12:00','16:00','20:00'],fontsize=12,)
plt.yticks(fontsize=12)
plt.savefig(os.path.join(plotsDir,'net_load.png'))


plt.subplot(312)
print("A1 shape:",A1.shape)
print("x1 shape:",x1.shape)
plt.plot(t1,(A1@x1.value),linewidth=2.0,label='Estimated load',color='g',alpha=0.6)
plt.plot(t1,Res_gt_days,linewidth=2.0,label='Ground Truth load',color='r',alpha=0.6)
plt.legend()
#plt.xlabel('Time',fontsize=14,**csfont)
plt.ylabel('Active Power (MW)',fontsize=12,**csfont)
#plt.title('Disaggregation Results',fontsize=14, **csfont)
plt.xticks([0, 8-1, 16-1,24-1,32-1,40-1], ['0:00', '4:00', '8:00','12:00','16:00','20:00'],fontsize=12,)
plt.yticks(fontsize=12)

#lt.show(311)

plt.subplot(313)
plt.plot(t1,A2@x2.value,linewidth=2.0,label='Estimated load',color='g',alpha=0.6)
plt.plot(t1,-Solar_gt_days,linewidth=2.0,label='Ground Truth load',color='r',alpha=0.6)
plt.legend()
plt.xlabel('Time',fontsize=14,**csfont)
plt.ylabel('Active Power (MW)',fontsize=12,**csfont)
#plt.title('Disaggregation Results',fontsize=14, **csfont)
plt.xticks([0, 8-1, 16-1,24-1,32-1,40-1], ['0:00', '4:00', '8:00','12:00','16:00','20:00'],fontsize=12,)
plt.yticks(fontsize=12)
plt.savefig(os.path.join(plotsDir,'disaggregation_results.png'))
plt.show()

plt.figure()
plt.plot(t1,A1@x1.value,label='Estimated load',color='b',alpha=0.7)
plt.show()

plt.figure()
plt.plot(t1,A2@x2.value,label='Estimated load')
plt.show()

#show aggregate ground turht
plt.figure(figsize=(10,5))
for i, array in Res_gt.items():
    plt.plot(t1, array,linewidth=2.0 )
#plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.xlabel('Time',fontsize=18,**csfont)
plt.ylabel('Active Power (MW)',fontsize=18,**csfont)
plt.title('300 Houses Residential Data',fontsize=18, **csfont)
plt.xticks([0, 8-1, 16-1,24-1,32-1,40-1], ['0:00', '4:00', '8:00','12:00','16:00','20:00'],fontsize=14,)
plt.yticks(fontsize=14)
plt.savefig(os.path.join(plotsDir,'300_houses_residential_data.png'))
plt.show()

plt.figure(figsize=(10,5))
for i, array in Solar_gt.items():
    plt.plot(t1, -array,linewidth=2.0 )
#plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.xlabel('Time',fontsize=18,**csfont)
plt.ylabel('Active Power (MW)',fontsize=18,**csfont)
plt.title('300 Houses Solar Data',fontsize=18, **csfont)
plt.xticks([0, 8-1, 16-1,24-1,32-1,40-1], ['0:00', '4:00', '8:00','12:00','16:00','20:00'],fontsize=14,)
plt.yticks(fontsize=14)
plt.savefig(os.path.join(plotsDir,'300_houses_solar_data.png'))
plt.show()
