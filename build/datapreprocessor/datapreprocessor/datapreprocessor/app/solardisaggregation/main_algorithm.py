# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import time
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import random
import cvxpy as cvx
import copy
import pickle
import os

start_time = time.time()
#data=pd.read_csv("ausgrid.csv")#read the dataset
datafile = os.path.join("/home/splathottam/GitHub/oedi/data/solardisaggregation/","ausgrid.csv")
data=pd.read_csv(datafile)#read the dataset

print("pd.read_csv took %s seconds" % (time.time() - start_time))

idd2=list(data.iloc[1:,0])

data2=data.iloc[1:,:]

category=list(data.iloc[1:,3])
itime_o=data.iloc[1:,4]
itime2= [None] * data.shape[0]
itime3= [None] * data.shape[0]
i=0
for time1 in itime_o:
    itime_temp=datetime.strptime(time1, '%d/%m/%Y')
    itime_temp2=itime_temp.strftime('%m/%d/%Y')
    itime2[i]=itime_temp
    itime3[i] = itime_temp2
    i=i+1

itime=itime2




idd=list(idd2.copy())
integer_map = map(int, idd)
idd_list = list(integer_map)

id_num=list(set(idd_list))

house=set(idd_list)
timesheet=[None]*len(id_num)
i=0
for x in timesheet:
    timesheet[i]={}
    i=i+1

Data_matrix=dict(zip(house,timesheet)).copy()

Data_matrix2 = copy.deepcopy(Data_matrix)
# create a dictionary which contains all the data for each house
for i in range(len(idd)-2):
    if itime[i] == itime[i+1] == itime[i+2]:
        index = idd_list[i]
        CL1 = data2.iloc[i,5:53].to_numpy()
        CL = CL1.astype(np.float)

        GC1 = data2.iloc[i+1,5:53].to_numpy()
        GC = GC1.astype(np.float)

        GG1 = data2.iloc[i+2,5:53].to_numpy()
        GG = GG1.astype(np.float)

        Agg1 = CL.copy() + GC.copy()
        Agg2 = Agg1.copy() - GG.copy()
        temp = [None,None,None,None, None]
        temp = [CL.copy(),GC.copy(),GG.copy(),Agg1.copy(),Agg2.copy()]
       # temp = [idd[i:i + 96].copy(), itime2[i:i + 96].copy(), agg[i:i + 96].copy(), Res_temp, solar[i:i + 96].copy()]
        Data_matrix[index][itime3[i]]=temp.copy()

for i in range(len(idd)-2):
    if itime[i] == itime[i+1] == itime[i+2]:
        index = idd_list[i]
        CL1 = data2.iloc[i,5:53].to_numpy()
        CL = CL1.astype(np.float)

        GC1 = data2.iloc[i+1,5:53].to_numpy()
        GC = GC1.astype(np.float)

        GG1 = data2.iloc[i+2,5:53].to_numpy()
        GG = GG1.astype(np.float)

        Agg1 = CL.copy() + GC.copy()
        Agg2 = Agg1.copy() - GG.copy()
        temp = [None,None,None]
        temp = [Agg1.copy(),GG.copy(), Agg2.copy()]
       # temp = [idd[i:i + 96].copy(), itime2[i:i + 96].copy(), agg[i:i + 96].copy(), Res_temp, solar[i:i + 96].copy()]
        Data_matrix2[index][itime3[i]]=temp.copy()
    elif category[i] == 'GC' and category[i+1] == 'GG' and category[i-1] != 'CL':
        index = idd_list[i]
        GC1 = data2.iloc[i, 5:53].to_numpy()
        GC = GC1.astype(np.float)

        GG1 = data2.iloc[i + 1, 5:53].to_numpy()
        GG = GG1.astype(np.float)

        Agg1 = GC.copy()
        Agg2 = Agg1.copy() - GG.copy()
        temp = [None, None, None]
        temp = [GC.copy(), GG.copy(), Agg2.copy()]
        # temp = [idd[i:i + 96].copy(), itime2[i:i + 96].copy(), agg[i:i + 96].copy(), Res_temp, solar[i:i + 96].copy()]
        Data_matrix2[index][itime3[i]] = temp.copy()


#select needed days from the start date
time_range = pd.date_range(datetime(2012, 7, 5), periods=11).to_pydatetime().tolist()

timelist=[temp.strftime('%m/%d/%Y') for temp in time_range]

Data_matrix_o=Data_matrix2[4]

#bb=[Data_matrix1[x][2] for x in timelist]


observed_agg={}
observed_res={}
observed_solar={}
id_num_ob=id_num[1:2+29]
#prepare the observed data
for timeday in timelist:
        #observed_agg.append(Data_matrix_o[x][2])
        #observed_res.append(Data_matrix_o[x][0])
        #observed_solar.append(Data_matrix_o[x][1])
        ob_aggregate = [0] * 48
        ob_Res = [0] * 48
        ob_Solar = [0] * 48
        for id in id_num_ob:
            if timeday in Data_matrix2[id]:
                bb = sum(np.isnan(list(Data_matrix2[id][timeday][2]))) == 0
                cc = sum(np.isnan(list(Data_matrix2[id][timeday][1]))) == 0
                dd = sum(np.isnan(list(Data_matrix2[id][timeday][0]))) == 0
                d = bb and cc and dd
                if d:
                    ob_aggregate = np.array(Data_matrix2[id][timeday][2]) + np.array(ob_aggregate)
                    ob_Res = np.array(Data_matrix2[id][timeday][0]) + np.array(ob_Res)
                    ob_Solar = np.array(Data_matrix2[id][timeday][1]) + np.array(ob_Solar)
                else:
                    print('NaN id and day are', id, timeday)
        observed_agg[timeday] = ob_aggregate
        observed_res[timeday] = ob_Res
        observed_solar[timeday] = ob_Solar

observed_res2=list(observed_res.values())
observed_solar2=list(observed_solar.values())
ob_res= np.transpose(np.array(observed_res2))
ob_solar = np.transpose(np.array(observed_solar2))
#Generate aggregate data and load
time_range = pd.date_range(datetime(2012, 7, 15), periods=260).to_pydatetime().tolist()

timelist2=[temp.strftime('%m/%d/%Y') for temp in time_range]

aggregate_t={}
Res_gt={}
Solar_gt={}
# prepare the data aggregated from 300 houses
aggregate_t1=[]
Res_gt1=[]
Solar_gt1=[]
j=0
for timeday in timelist2:
    aggregate=[0]*48
    Res=[0]*48
    Solar=[0]*48
    for id in id_num:
        if timeday in Data_matrix2[id]:
            bb=sum(np.isnan(list(Data_matrix2[id][timeday][2]))) ==0
            cc=sum(np.isnan(list(Data_matrix2[id][timeday][1]))) == 0
            dd=sum(np.isnan(list(Data_matrix2[id][timeday][0]))) == 0
            d=bb and cc and dd
            if d:
                aggregate = np.array(Data_matrix2[id][timeday][2]) + np.array(aggregate)
                Res = np.array(Data_matrix2[id][timeday][0]) + np.array(Res)
                Solar= np.array(Data_matrix2[id][timeday][1]) + np.array(Solar)
            else:
                print('NaN id and day are', id, timeday)
    aggregate_t[timeday]=aggregate
    Res_gt[timeday] = Res
    Solar_gt[timeday] = Solar
    aggregate_t1.append(aggregate)
    Res_gt1.append(Res)
    Solar_gt1.append(Solar)
#save the dataset for DNN and LSTM model
df1={}
df2={}

df1['aggr']=[]
df1['res'] = []
df1['solar'] = []

df2['aggr']=[]
df2['res'] = []
df2['solar'] = []


df1['aggr']=aggregate_t1[:240]
df1['res'] = Res_gt1[:240]
df1['solar'] = Solar_gt1[:240]

df2['aggr']=aggregate_t1[241:]
df2['res'] = Res_gt1[241:]
df2['solar'] = Solar_gt1[241:]

train = df1
val = df2

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


print('finished data proparation')

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
plt.savefig('figure.png')
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
plt.savefig('figure2.png')
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
prob = cvx.Problem(objective,constraints)
print("Optimal value", prob.solve())
print("Optimal var")
print('x1=', x1.value)
print('x2=', x2.value)

error1=cvx.norm2(A1@x1-Res_gt_days).value/cvx.norm2(Res_gt_days).value
error2=cvx.norm2(-A2@x2-Solar_gt_days).value/cvx.norm2(Solar_gt_days).value

print("error1=", error1)
print("error2=", error2)

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


plt.subplot(312)
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
plt.savefig('figure3.png')
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
plt.savefig('figure4.png')
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
plt.savefig('figure5.png')
plt.show()
