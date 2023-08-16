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



start_time = time.time()
data=pd.read_csv("austin.csv")#read the dataset

print("pd.read_csv took %s seconds" % (time.time() - start_time))

idd2=data.iloc[:,0]

itime_o=data.iloc[:,1]
itime2= [None] * data.shape[0]
itime3= [None] * data.shape[0]
i=0
for time1 in itime_o:
    itime_temp=datetime.strptime(time1, '%m/%d/%Y %H:%M')
    itime_temp2=itime_temp.strftime('%m/%d/%Y')
    itime2[i]=itime_temp
    itime3[i] = itime_temp2
    i=i+1

itime=itime2


agg_o=data.iloc[:,31]


idd=idd2.copy()
agg=agg_o.copy()

solar_o=data.iloc[:,67]

solar=solar_o.copy()


id_num=list(set(idd))

house=set(idd)
timesheet=[None]*len(id_num)
i=0
for x in timesheet:
    timesheet[i]={}
    i=i+1

Data_matrix=dict(zip(house,timesheet))
i=0
# create a dictionary which contains all the data for each house
for i in range(len(itime)-96):
    if itime[i].hour == 00 and itime[i].minute == 00 and itime[i+96].hour==00 and itime[i+96].minute == 00 and len(set(idd[i:i+96]))==1:
        index = idd[i]
        noise1=np.random.normal(0,0.2,96).copy()
        noise2 = np.random.normal(0, 0.1, 96).copy()
        Res_temp = agg[i:i+96].copy()+solar[i:i+96].copy()
        Res_temp[40:70:3] = Res_temp[40:70:3].copy() + noise1[40:70:3].copy()
        Solar_temp= solar[i:i+96].copy()
        Solar_temp[40:70:3] =  Solar_temp[40:70:3].copy()+noise2[40:70:3].copy()
        Agg_temp=Res_temp-Solar_temp
        temp=[None,None,None,None, None]
        temp=[idd[i:i+96].copy(),itime2[i:i+96].copy(), Agg_temp,Res_temp, Solar_temp]
       # temp = [idd[i:i + 96].copy(), itime2[i:i + 96].copy(), agg[i:i + 96].copy(), Res_temp, solar[i:i + 96].copy()]
        Data_matrix[index][itime3[i]]=temp.copy()
    i=i+1



time_range = pd.date_range(datetime(2018, 1, 1), periods=11).to_pydatetime().tolist()

timelist=[temp.strftime('%m/%d/%Y') for temp in time_range]

Data_matrix1=Data_matrix[661]

#bb=[Data_matrix1[x][2] for x in timelist]


observed_agg=[]
observed_res=[]
observed_solar=[]
#prepare the observed data
for x in timelist:
    if sum(np.isnan(Data_matrix1[x][2]))==0 and sum(np.isnan(Data_matrix1[x][3]))==0 and sum(np.isnan(Data_matrix1[x][4]))==0:
        observed_agg.append(Data_matrix1[x][2])
        observed_res.append(Data_matrix1[x][3])
        observed_solar.append(Data_matrix1[x][4])
    else:
        print('Nan day',x)

ob_res= np.transpose(np.array(observed_res))
ob_solar = np.transpose(np.array(observed_solar))
#Generate aggregate data and load
time_range = pd.date_range(datetime(2018, 1, 11), periods=15).to_pydatetime().tolist()

timelist2=[temp.strftime('%m/%d/%Y') for temp in time_range]

aggregate_t={}
Res_gt={}
Solar_gt={}
# prepare the data aggregated from 25 houses
j=0
for timeday in timelist2:
    aggregate=[0]*96
    Res=[0]*96
    Solar=[0]*96
    for id in id_num:
        if timeday in Data_matrix[id]:
            bb=sum(np.isnan(list(Data_matrix[id][timeday][2]))) ==0
            cc=sum(np.isnan(list(Data_matrix[id][timeday][3]))) == 0
            dd=sum(np.isnan(list(Data_matrix[id][timeday][4]))) == 0
            d=bb and cc and dd
            if d:
                aggregate = np.array(Data_matrix[id][timeday][2]) + np.array(aggregate)
                Res = np.array(Data_matrix[id][timeday][3]) + np.array(Res)
                Solar= np.array(Data_matrix[id][timeday][4]) + np.array(Solar)
            else:
                print('NaN id and day are', id, timeday)
    aggregate_t[timeday]=aggregate
    Res_gt[timeday] = Res
    Solar_gt[timeday] = Solar

selectday='01/20/2018'
t1 = np.arange(0.0, 96.0, 2)
t1=range(96)
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
t1=range(96)
plt.figure(figsize=(10,5))
for i, array in enumerate(np.transpose(A1)):
    plt.plot(t1, array,linewidth=2.0 )
#plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.xlabel('Time',fontsize=18,**csfont)
plt.ylabel('Active Power (MW)',fontsize=18,**csfont)
plt.title('Observed Residential Data',fontsize=18, **csfont)
plt.xticks([0, 16-1, 32-1,48-1,64-1,80-1], ['0:00', '4:00', '8:00','12:00','16:00','20:00'],fontsize=14,)
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
plt.xticks([0, 16-1, 32-1,48-1,64-1,80-1], ['0:00', '4:00', '8:00','12:00','16:00','20:00'],fontsize=14,)
plt.yticks(fontsize=14)
plt.savefig('figure2.png')
plt.show()


start_time = time.time()
#wdata=pd.read_csv("weather2.csv")
#read weather data in austin
wdata = pd.read_csv("weather2.csv", parse_dates=[['Year','Month', 'Day','Hour','Minute']])


print("pd.read_csv took %s seconds" % (time.time() - start_time))

Time_data=wdata.iloc[:,0]
#DHI DNI GHI are the irradiance index wind and temp are the indices for wind and temperature
Data_DHI=wdata.iloc[:,1]
Data_DNI=wdata.iloc[:,2]
Data_GHI=wdata.iloc[:,3]

Data_wind=wdata.iloc[:,4]
Data_temp=wdata.iloc[:,5]

wtime2= [None] * wdata.shape[0]
wtime3= [None] * wdata.shape[0]

i=0
for time2 in Time_data:
    itime_temp=datetime.strptime(time2, '%Y %m %d %H %M')
    itime_temp2=itime_temp.strftime('%m/%d/%Y')
    wtime2[i]=itime_temp
    wtime3[i] = itime_temp2
    i=i+1


#itime=sorted(itime2)


#index_t = sorted(range(len(itime2)), key=lambda k: itime2[k])
#index_t = sorted(range(len(itime2)), key=lambda k: itime[k])

#itime4=[itime3[i] for i in index_t]


day_num=set(wtime3)
timesheet=[None]*len(day_num)
wData_matrix={}

#data.shape[0]
i=0

for i in range(len(wtime2)-48):
    if wtime2[i].hour == 00 and wtime2[i].minute == 00 and wtime2[i+48].hour==00 and wtime2[i+48].minute == 00:

        temp=[None,None,None,None, None]
        temp=[Data_DHI[i:i+48].copy(),Data_DNI[i:i+48].copy(),Data_GHI[i:i+48].copy(),Data_wind[i:i+48], Data_temp[i:i+48].copy()]
        wData_matrix[wtime3[i]]=temp.copy()
    i=i+1

day2='01/11/2018'
t1 = np.arange(0.0, 48.0, 2)
t1=range(48)
csfont = {'fontname':'Times New Roman'}
plt.figure(figsize=(8,8))
plt.subplot(311)
plt.plot(t1,wData_matrix[day2][0])
plt.xticks([0, 8-1, 16-1,24-1,32-1,40-1], ['0:00', '4:00', '8:00','12:00','16:00','20:00'],fontsize=14,)
plt.yticks(fontsize=14)
#plt.xlabel('Time',fontsize=18,**csfont)
plt.ylabel('DHI',fontsize=18,**csfont)
#lt.show(311)

plt.subplot(312)
plt.plot(t1,wData_matrix[day2][1])
plt.xticks([0, 8-1, 16-1,24-1,32-1,40-1], ['0:00', '4:00', '8:00','12:00','16:00','20:00'],fontsize=14,)
plt.yticks(fontsize=14)
#plt.xlabel('Time',fontsize=18,**csfont)
plt.ylabel('DNI',fontsize=18,**csfont)
#plt.show()

plt.subplot(313)
plt.plot(t1,wData_matrix[day2][2])
plt.xticks([0, 8-1, 16-1,24-1,32-1,40-1], ['0:00', '4:00', '8:00','12:00','16:00','20:00'],fontsize=14,)
plt.yticks(fontsize=14)
plt.xlabel('Time',fontsize=18,**csfont)
plt.ylabel('GHI',fontsize=18,**csfont)


#plt.subplot(414)
#plt.plot(t1,wData_matrix[day2][4])
#plt.xticks([0, 8-1, 16-1,24-1,32-1,40-1], ['0:00', '4:00', '8:00','12:00','16:00','20:00'],fontsize=14,)
#plt.yticks(fontsize=14)
#plt.xlabel('Time',fontsize=18,**csfont)
#plt.ylabel('Temperature',fontsize=18,**csfont)
plt.savefig('figure_w.png')
plt.show()


print('Data_matrix first column', wData_matrix['01/01/2018'][0] )
print('Data_matrix second column', wData_matrix['01/01/2018'][1] )
print('Data_matrix third column', wData_matrix['01/01/2018'][2] )
print('Data_matrix fourth column', wData_matrix['01/01/2018'][4] )

tempp=wData_matrix[day2]
temp22=wData_matrix[day2][0]
temp33=wData_matrix[day2][1]
w_feature=[wData_matrix['01/11/2018'][0], wData_matrix['01/11/2018'][1],wData_matrix['01/11/2018'][2] ]

wData1=[]
wData2=[]
wData3=[]

for i in range(96):
    if i%2==0:
        wtemp1 = wData_matrix[day2][0].to_numpy()
        wtemp2 = wData_matrix[day2][1].to_numpy()
        wtemp3 = wData_matrix[day2][2].to_numpy()
        wData1.append(wtemp1[int(i/2)])
        wData2.append(wtemp2[int(i/2)])
        wData3.append(wtemp3[int(i/2)])
    else:
        wData1.append(float("nan"))
        wData2.append(float("nan"))
        wData3.append(float("nan"))

wDatas1=pd.Series(wData1)
wDatass1=wDatas1.interpolate()
wDatas2=pd.Series(wData2).interpolate()
wDatass2=wDatas2.interpolate()
wDatas3=pd.Series(wData3).interpolate()
wDatass3=wDatas3.interpolate()

wtotl=np.transpose(np.array([wDatass1.to_numpy(),wDatass2.to_numpy(),wDatass3.to_numpy()]))
#A3 contains the weather information and the observed patterns in A2
A3=np.concatenate((wtotl, A2), axis=1)
#A3=wtotl
[m, K1]=A1.shape
[m, K2]=A2.shape

selectday2='01/11/2018'
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

#K2=0
x3 = cvx.Variable((K1,n))
#x4 = cvx.Variable((3,n))
x4 = cvx.Variable((K2+3,n))

objective = cvx.Minimize(cvx.sum_squares(bb-(A1@x1+A2@x2))+0.5*cvx.norm1(x1)+0.5*cvx.norm1(x2))
#constraints = [0 <= x1, x2 <= 0]
prob = cvx.Problem(objective)
print("Optimal value2", prob.solve())
objective2 = cvx.Minimize(cvx.sum_squares(bb-(A1@x3+A3@x4))+1*cvx.norm1(x3)+4*cvx.norm1(x4))
#constraints = [0 <= x1, x2 <= 0]
prob2 = cvx.Problem(objective2)
print("Optimal value2", prob2.solve())
print("Optimal var2")
print(x1.value)
print(x2.value)

temp=A1@x1
error1=cvx.norm2(A1@x1-Res_gt_days).value/cvx.norm2(Res_gt_days).value
error2=cvx.norm2(-A2@x2-Solar_gt_days).value/cvx.norm2(Solar_gt_days).value

error3=cvx.norm2(A1@x3-Res_gt_days).value/cvx.norm2(Res_gt_days).value
error4=cvx.norm2(-A3@x4-Solar_gt_days).value/cvx.norm2(Solar_gt_days).value

print("error1=", error1)
print("error2=", error2)
print("error3=", error3)
print("error4=", error4)

tempp=np.transpose(A1@x1.value)
t1 = np.arange(0.0, 96.0, 2)
t1=range(96)
plt.figure(figsize=(8,6))
plt.subplot(311)
plt.plot(t1,bb,linewidth=2.0,label='Net Load',color='b',alpha=0.6)
plt.legend()
#plt.xlabel('Time',fontsize=14,**csfont)
plt.ylabel('Active Power (MW)',fontsize=12,**csfont)
#plt.title('Net Load',fontsize=14, **csfont)
plt.xticks([0, 16-1, 32-1,48-1,64-1,80-1], ['0:00', '4:00', '8:00','12:00','16:00','20:00'],fontsize=12,)
plt.yticks(fontsize=12)


plt.subplot(312)
plt.plot(t1,(A1@x1.value),linewidth=2.0,label='Estimated load',color='g',alpha=0.6)
plt.plot(t1,Res_gt_days,linewidth=2.0,label='Ground Truth load',color='r',alpha=0.6)
plt.legend()
#plt.xlabel('Time',fontsize=14,**csfont)
plt.ylabel('Active Power (MW)',fontsize=12,**csfont)
#plt.title('Disaggregation Results',fontsize=14, **csfont)
plt.xticks([0, 16-1, 32-1,48-1,64-1,80-1], ['0:00', '4:00', '8:00','12:00','16:00','20:00'],fontsize=12,)
plt.yticks(fontsize=12)

#lt.show(311)

plt.subplot(313)
plt.plot(t1,A2@x2.value,linewidth=2.0,label='Estimated load',color='g',alpha=0.6)
plt.plot(t1,-Solar_gt_days,linewidth=2.0,label='Ground Truth load',color='r',alpha=0.6)
plt.legend()
plt.xlabel('Time',fontsize=14,**csfont)
plt.ylabel('Active Power (MW)',fontsize=12,**csfont)
#plt.title('Disaggregation Results',fontsize=14, **csfont)
plt.xticks([0, 16-1, 32-1,48-1,64-1,80-1], ['0:00', '4:00', '8:00','12:00','16:00','20:00'],fontsize=12,)
plt.yticks(fontsize=12)
plt.savefig('figure3.png')
plt.show()




plt.figure()
plt.plot(t1,A3@x4.value,linewidth=2.0,label='Estimated load',color='g',alpha=0.6)
plt.plot(t1,-Solar_gt_days,linewidth=2.0,label='Ground Truth load',color='r',alpha=0.6)
plt.legend()
plt.xlabel('Time',fontsize=14,**csfont)
plt.ylabel('Active Power (MW)',fontsize=12,**csfont)
#plt.title('Disaggregation Results',fontsize=14, **csfont)
plt.xticks([0, 16-1, 32-1,48-1,64-1,80-1], ['0:00', '4:00', '8:00','12:00','16:00','20:00'],fontsize=12,)
plt.yticks(fontsize=12)
plt.savefig('figure4.png')
plt.show()


plt.figure(figsize=(8,6))
plt.subplot(311)
plt.plot(t1,bb,linewidth=2.0,label='Net Load',color='b',alpha=0.6)
plt.legend()
#plt.xlabel('Time',fontsize=14,**csfont)
plt.ylabel('Active Power (MW)',fontsize=12,**csfont)
#plt.title('Net Load',fontsize=14, **csfont)
plt.xticks([0, 16-1, 32-1,48-1,64-1,80-1], ['0:00', '4:00', '8:00','12:00','16:00','20:00'],fontsize=12,)
plt.yticks(fontsize=12)


plt.subplot(312)
plt.plot(t1,(A1@x3.value),linewidth=2.0,label='Estimated load',color='g',alpha=0.6)
plt.plot(t1,Res_gt_days,linewidth=2.0,label='Ground Truth load',color='r',alpha=0.6)
plt.legend()
#plt.xlabel('Time',fontsize=14,**csfont)
plt.ylabel('Active Power (MW)',fontsize=12,**csfont)
#plt.title('Disaggregation Results',fontsize=14, **csfont)
plt.xticks([0, 16-1, 32-1,48-1,64-1,80-1], ['0:00', '4:00', '8:00','12:00','16:00','20:00'],fontsize=12,)
plt.yticks(fontsize=12)

#lt.show(311)

plt.subplot(313)
plt.plot(t1,A3@x4.value,linewidth=2.0,label='Estimated load',color='g',alpha=0.6)
plt.plot(t1,-Solar_gt_days,linewidth=2.0,label='Ground Truth load',color='r',alpha=0.6)
plt.legend()
plt.xlabel('Time',fontsize=14,**csfont)
plt.ylabel('Active Power (MW)',fontsize=12,**csfont)
#plt.title('Disaggregation Results',fontsize=14, **csfont)
plt.xticks([0, 16-1, 32-1,48-1,64-1,80-1], ['0:00', '4:00', '8:00','12:00','16:00','20:00'],fontsize=12,)
plt.yticks(fontsize=12)
plt.savefig('figure55.png')
plt.show()

#show aggregate ground turht
plt.figure(figsize=(10,5))
for i, array in Res_gt.items():
    plt.plot(t1, array,linewidth=2.0 )
#plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.xlabel('Time',fontsize=18,**csfont)
plt.ylabel('Active Power (MW)',fontsize=18,**csfont)
plt.title('25 Houses Residential Data',fontsize=18, **csfont)
plt.xticks([0, 16-1, 32-1,48-1,64-1,80-1], ['0:00', '4:00', '8:00','12:00','16:00','20:00'],fontsize=14,)
plt.yticks(fontsize=14)
plt.savefig('figure4.png')
plt.show()

plt.figure(figsize=(10,5))
for i, array in Solar_gt.items():
    plt.plot(t1, -array,linewidth=2.0 )
#plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.xlabel('Time',fontsize=18,**csfont)
plt.ylabel('Active Power (MW)',fontsize=18,**csfont)
plt.title('25 Houses Solar Data',fontsize=18, **csfont)
plt.xticks([0, 16-1, 32-1,48-1,64-1,80-1], ['0:00', '4:00', '8:00','12:00','16:00','20:00'],fontsize=14,)
plt.yticks(fontsize=14)
plt.savefig('figure5.png')
plt.show()
