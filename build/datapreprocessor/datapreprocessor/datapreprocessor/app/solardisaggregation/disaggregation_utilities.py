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

def get_data_matrix(datafile):
    """Create data matrix"""
    
    start_time = time.time()
    data=pd.read_csv(datafile)#read the dataset

    print(f"pd.read_csv took {time.time() - start_time:.2f} seconds")

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
    
    return Data_matrix_o,Data_matrix2,id_num,timelist

def get_observed_data(Data_matrix2,timelist,id_num_ob):
    """prepare and return the observed data"""
    
    observed_agg={}
    observed_res={}
    observed_solar={}

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
            
    return observed_agg,observed_res,observed_solar

def get_aggregate_data(Data_matrix2,id_num):
    """Generate aggregate data"""
    
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
    
    return bb,aggregate_t,Res_gt,Solar_gt,df1,df2
