"""'
Created on Tuesday August 01 15:00:00 2023
@author: Siby Plathottam
"""
import calendar
import time
import math
import os
from typing import List, Set, Dict, Tuple, Optional, Union

import numpy as np
import pandas as pd

from oedianl.app.nodeload.timeseries_data_utilities import get_time_series_dataframe,add_milliseconds

def get_pmu_data(pmu_file):
    
    df = pd.read_csv(pmu_file,nrows=10)#,index_col=0
    features = {"FDRID":"fdrid","TimeStamp":"datetime","Conv":"fractional_seconds"}
    if "Voltage         _Mag" in df.columns:
        features.update({"Voltage         _Mag":"voltage_mag"})
    if "Voltage         _Ang" in df.columns:
        features.update({"Voltage         _Ang":"voltage_angle"})    
    if 'Current         _Mag' in df.columns:
        features.update({"Current         _Mag":"current_mag"})
    if 'Current         _Ang' in df.columns:
        features.update({"Current         _Ang":"current_angle"})
    
    df = pd.read_csv(pmu_file,usecols= list(features.keys()))#,index_col=0
    df = df.rename(columns=features)    
    
    return df

def pmu_to_timeseries(pmu_file,datetime_column = "datetime",id_column = "fdrid",save_folder=""):
    """Convert pmu data to timeseries"""
    df = get_pmu_data(pmu_file)
    pmu_ids = list(df["fdrid"].unique())
    print(f"Found following {len(pmu_ids)} ids:{pmu_ids}")
    
    df = add_milliseconds(df,datetime_column="datetime")
    n_days = len(df["datetime"].dt.day.unique())
    df_timeseries = pd.DataFrame()
    
    df_timeseries[datetime_column] = df[datetime_column] 
    for i,pmu_id in enumerate(pmu_ids):
        df_timeseries[f"vmag_pmu_{pmu_id}_{i+1}"] = df[df[id_column]==pmu_id]["voltage_mag"].values
        df_timeseries[f"vang_pmu_{pmu_id}_{i+1}"] = df[df[id_column]==pmu_id]["voltage_angle"].values
        if "current_mag" in df.columns:
            df_timeseries[f"imag_pmu_{pmu_id}_{i+1}"] = df[df[id_column]==pmu_id]["current_mag"].values
        if "current_angle" in df.columns:
            df_timeseries[f"iang_pmu_{pmu_id}_{i+1}"] = df[df[id_column]==pmu_id]["current_angle"].values
    
    if save_folder:
        save_filepath = os.path.join(save_folder,f"pmu_n_pmu-{len(pmu_ids)}_n_days-{n_days}.csv")
        print(f"Saving pmu time series in {save_filepath}")
        df_timeseries.to_csv(path_or_buf =save_filepath,index_label='block_index')
    
    return df_timeseries

def generate_node_voltage_profiles(df_voltage,n_days=1,start_year = 2016,start_month=1,start_day=1):
	"""Generate volta"""
		
	print(f"Creating node voltage profiles for year:{start_year},month:{start_month}")
	
	df_node_voltage = pd.DataFrame()
	
	n_timesteps_per_day = len(df_voltage)
	print(f"Number of time steps per day:{n_timesteps_per_day}")
	time_interval_in_minutes = f"{int((24*60)/n_timesteps_per_day)} min" #"30min" #Calculating time interval in minutes
	print(f"Time interval in minutes:{time_interval_in_minutes}")
	time_interval_in_minutes = f"{int((df_voltage['datetime'].iloc[-1]-df_voltage['datetime'].iloc[-2]).total_seconds()/60.0)} min"	
	print(f"Time interval in minutes:{time_interval_in_minutes}")
	
	pmu_ids = get_unique_measurement_ids_in_df(df,measurement_types = ["vmag_pmu_","vang_pmu_"])
	print(f"Found following {len(pmu_ids)} pmu IDs:{pmu_ids}")
	node_voltage_dict = {f"node_{i+1}":df_voltage[f"vmag_pmu_{pmu_id}"].values for i,pmu_id in enumerate(pmu_ids)} #Assign a pmu id to each node        

	df_node_voltage = pd.DataFrame.from_dict(node_voltage_dict) #Faster
	df_node_voltage.insert(0,'datetime',df_voltage["datetime"]) #Insert at first columns
	
	return df_node_voltage,node_voltage_dict

def get_unique_measurement_ids_in_df(df,measurement_types):
	selected_columns = [col for col in df.columns if any(col.startswith(prefix) for prefix in measurement_types)]
	pmu_ids = []
	for col in df[selected_columns].columns:
		for measurement_type in measurement_types:
			if measurement_type in col:
				pmu_ids.append(col.replace(measurement_type,""))
	return list(set(pmu_ids))
