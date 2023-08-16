"""'
Created on Auguest 01 00:00:00 2022
@author: Argonne National Laboratory
"""

import time
import random
import copy
import pickle
import os
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cvx

from disaggregation_utilities import get_data_matrix,get_observed_data,get_aggregate_data

datafile = os.path.join("/home/splathottam/GitHub/oedi/data/solardisaggregation/","ausgrid.csv")
Data_matrix_o,Data_matrix2,id_num,timelist = get_data_matrix(datafile)

id_num_ob=id_num[1:2+29]
observed_agg,observed_res,observed_solar = get_observed_data(Data_matrix2,timelist,id_num_ob)

observed_res2=list(observed_res.values())
observed_solar2=list(observed_solar.values())
ob_res= np.transpose(np.array(observed_res2))
ob_solar = np.transpose(np.array(observed_solar2))

bb,aggregate_t,Res_gt,Solar_gt,df1,df2 = get_aggregate_data(Data_matrix2,id_num)

print("bb",)

train = df1
val = df2bb

#save('data/train2.npy', train)
#save('data/val2.npy', val)
# here save the training and validation dataset for deep learning models.
a_file = open("train_aus.pkl", "wb")
pickle.dump(train, a_file)
a_file.close()

b_file = open("val_aus.pkl", "wb")
pickle.dump(val, b_file)
b_file.close()

#select one day and visualize the data
selectday='07/15/2012'
t1 = np.arange(0.0, 48.0, 2)
t1=range(48)
plt.figure()
plt.subplot(311)
plt.plot(t1,aggregate_t[selectday])
plt.xlabel('Time')
plt.ylabel('total aggregate Data')
#lt.show(311)

plt.subplot(312)
plt.plot(t1,Res_gt[selectday])
plt.xlabel('Time')
plt.ylabel('total residential Data')
#plt.show()

plt.subplot(313)
plt.plot(t1,Solar_gt[selectday])
plt.xlabel('Time')
plt.ylabel('total Solar Data')
plt.show()


print('finished data preparation')

A1 = ob_res.copy()
A2 = ob_solar.copy()
csfont = {'fontname':'Times New Roman'}
t1=range(48)
plt.figure(figsize=(10,5))
for i, array in enumerate(np.transpose(A1)):
    plt.plot(t1, array,linewidth=2.0 )
#plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.xlabel('Time',fontsize=18,**csfont)
plt.ylabel('Active Power (MW)',fontsize=18,**csfont)
plt.title('Observed Residential Data',fontsize=18, **csfont)
plt.xticks([0, 8-1, 16-1,24-1,32-1,40-1], ['0:00', '4:00', '8:00','12:00','16:00','20:00'],fontsize=14,)
plt.yticks(fontsize=14)
plt.savefig('observed_residential_data.png')
plt.show()

plt.figure(figsize=(10,5))
for i, array in enumerate(np.transpose(A2)):
    plt.plot(t1, -array,linewidth=2.0 )
#plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.xlabel('Time',fontsize=18,**csfont)
plt.ylabel('Active Power (MW)',fontsize=18,**csfont)
plt.title('Observed Solar Data',fontsize=18, **csfont)
plt.xticks([0, 8-1, 16-1,24-1,32-1,40-1], ['0:00', '4:00', '8:00','12:00','16:00','20:00'],fontsize=14,)
plt.yticks(fontsize=14)
plt.savefig('observed_solar_data.png')
plt.show()

[m, K1]=A1.shape
[m, K2]=A2.shape

selectday2='07/20/2012'
b = np.array( [aggregate_t[selectday2]])
Res_gt_day=np.array([Res_gt[selectday2]])
Solar_gt_day=[np.transpose(Solar_gt[selectday2])]
bb=np.transpose(b)
Res_gt_days = np.transpose(Res_gt_day)
Solar_gt_days = np.transpose(Solar_gt_day)
n=1
# Construct the problem. solve the problem by lasso
x1 = cvx.Variable((K1,n))
x2 = cvx.Variable((K2,n))
objective = cvx.Minimize(cvx.sum_squares(bb-(A1@x1+A2@x2))+1*cvx.norm1(x1)+1*cvx.norm1(x2))
constraints = [x1>=0, x2 <= 0]
#constraints = []

print("Solving lasso regression least squares optimization problem...")
prob = cvx.Problem(objective,constraints)
print("Optimal value", prob.solve())
print("Optimal var")
print('x1=', x1.value)
print('x2=', x2.value)

error1=cvx.norm2(A1@x1-Res_gt_days).value/cvx.norm2(Res_gt_days).value
error2=cvx.norm2(-A2@x2-Solar_gt_days).value/cvx.norm2(Solar_gt_days).value

print("error1=", error1)
print("error2=", error2)

lasso_regression_weights = {'x1': x1,'x2': x2}

print("Saving x2 and x2 in lasso_regression_weights.pickle")
with open('lasso_regression_weights.pickle', 'wb') as handle:
    pickle.dump(lasso_regression_weights, handle, protocol=pickle.HIGHEST_PROTOCOL)

