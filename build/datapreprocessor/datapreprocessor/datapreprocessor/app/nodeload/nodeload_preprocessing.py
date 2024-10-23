"""'
Created on Thursday Feb 26 15:00:00 2023
@author: Siby Plathottam
"""
import calendar
import math
from typing import List, Set, Dict, Tuple, Optional, Union

import numpy as np
import pandas as pd

from datapreprocessor.app.nodeload.timeseries_data_utilities import get_time_series_dataframe

def get_df_from_timeseries_file(time_series_files,load_type = "",load_block_length = 4,selected_month=1,n_timesteps_per_day=48,show_details=True,cyclical_features=[],upsample=False,upsample_time_period="15Min"):
	"""Create a dataframe from one or more time series CSV files"""
	
	df_train = pd.DataFrame()
	
	datetimes = []
	load_values =[]
	load_ids = []
	zip_codes = []
	unique_zip_codes = []
	time_series_file_with_NAN = []
	time_series_file_without_loadtype = []
	time_series_file_without_month = []
	time_series_files_used = []
	
	expected_number_of_timestamps_per_file = calendar.monthrange(2018,selected_month)[1]*n_timesteps_per_day  #len(df_temp)
	
	total_load_blocks = int(expected_number_of_timestamps_per_file/load_block_length)
	if show_details:
		print(f"Total load blocks:{total_load_blocks}")
	assert total_load_blocks*load_block_length ==  expected_number_of_timestamps_per_file, "Load block length should be a multiple of total time steps in file"
		
	load_type_count = 0
	time_series_file_count = 0
	
	for time_series_file in time_series_files:
		if show_details:
			print(f"Reading file:{time_series_file}")
		df_raw = get_time_series_dataframe(time_series_file)
		zip_code =	int(time_series_file.split("/")[-1][8:13])	   
		
		if df_raw.isnull().values.any():
			print(f"Null value found in file:{time_series_file}....skipping")
			time_series_file_with_NAN.append(time_series_file)
		else:
			load_type_col_names = [col_name for col_name in list(df_raw.keys()) if load_type in col_name.lower()]
			if len(load_type_col_names)>0:
				df_raw['time_block'] = df_raw['time_block'].shift(1) #Shift to fix timestamp hour format
				#df_raw['time_block'][0] = '00:00:00' #Fill missing value due to shift
				df_raw.loc[0,'time_block'] = '00:00:00' #Fill missing value due to shift
				
				if "datetime" not in df_raw.columns:
					df_raw.insert(0, "datetime", df_raw['date_block'] + "-" + df_raw['time_block']) #Insert date time block
				else:
					df_raw["datetime"] = df_raw['date_block'] + "-" + df_raw['time_block'] #Insert date time block
				
				df_raw['datetime'] = df_raw['datetime'].str.replace("-24:","-00:")
				df_raw['datetime']	= pd.to_datetime(df_raw['datetime'], format='%Y-%m-%d-%H:%M:%S')
				
				if upsample:
					df_raw = get_upsampled_df(df_timeseries,datetime_column="datetime",upsample_time_period=upsample_time_period)
				
				found_month = set(df_raw['datetime'].dt.month.to_list())
				
				assert len(found_month) == 1, "Only one month is expected per file"
				if selected_month == list(found_month)[0]:
					if show_details:
						print(f"Found {len(load_type_col_names)} load types:{load_type} in month:{found_month}")
					number_of_timestamps = len(df_raw) 
					assert expected_number_of_timestamps_per_file == number_of_timestamps, f"{time_series_file} does not have {expected_number_of_timestamps_per_file} timestamps as expected!"			  

					load_type_count = load_type_count + len(load_type_col_names)
					time_series_file_count = time_series_file_count+1
					datetimes.extend(list(df_raw['datetime'].values)*len(load_type_col_names))
					load_values.extend(list(np.concatenate([df_raw[load_type_col_name] for load_type_col_name in load_type_col_names]).flat))
					#load_ids.extend(list(np.concatenate([[load_type_col_name]*len(df_raw['datetime']) for load_type_col_name in load_type_col_names]).flat))
					load_ids.extend(list(np.concatenate([[f"{load_type}_{time_series_file_count}_{i}"]*len(df_raw['datetime']) for i,load_type_col_name in enumerate(load_type_col_names)]).flat))
					zip_codes.extend([zip_code]*len(df_raw['datetime'])*len(load_type_col_names))
					unique_zip_codes.append(zip_code)
					time_series_files_used.append(time_series_file)
				else:
					time_series_file_without_month.append(time_series_file)
					if show_details:
						print(f"File:{time_series_file} contains data from month {found_month}...skipping")
			else:
				time_series_file_without_loadtype.append(time_series_file)
				if show_details:
					print(f"Could not find load type:{load_type} in {time_series_file}")
	
	df_train["datetime"]  = datetimes
	df_train["load_value"]	= load_values
	df_train["load_id"]	 = load_ids
	df_train["zip_code"]  = zip_codes
	
	assert load_type_count*expected_number_of_timestamps_per_file == len(df_train["datetime"]), f"Number of timesteps in df_train:{len(df_train['datetime'])} not matching with expected"

	df_train['datetime']  = pd.to_datetime(df_train['datetime'], format='%Y-%m-%d-%H:%M:%S')
	
	if show_details:
		print(f"Found following unique zip codes:{set(unique_zip_codes)}")
		print(f"Total files without selected load type:{load_type} - {len(time_series_file_without_loadtype)}")
		print(f"Total files without selected month:{selected_month} - {len(time_series_file_without_month)}")
		print(f"File withs NAN:{time_series_file_with_NAN}")
		print(f"Total files used in dataframe:{len(time_series_files_used)}")
		print(f"Dataframe contains load types:{load_type_count} with total length:{len(df_train['datetime'])}")
	
	df_train = encode_cyclical_features(df_train,cyclical_features,show_df=False,show_plot=False)
	
	return df_train

def encode_cyclical_features(df,cyclical_features:List[str]=[],show_df=False,show_plot=False):
	"""Encode cyclical datetime features using cos-sin encoding"""
	
	for cyclical_feature in cyclical_features:
		print(f"Encoding cyclical feature:{cyclical_feature}")
		if cyclical_feature == 'hour_of_day':
			#df['hour']=df.index.hour
			df['hour']=df["datetime"].dt.hour
			df["hour_norm"] = 2 * math.pi * df["hour"] / 23.0 #df["hour"].max() # We normalize x values to match with the 0-2π cycle
			df["cos_hour"] = np.cos(df["hour_norm"])
			df["sin_hour"] = np.sin(df["hour_norm"])
			if show_plot:
				df.plot.scatter(x="cos_hour", y="sin_hour").set_aspect('equal')
			df = df.drop(columns=['hour_norm'])
		
		elif cyclical_feature == 'day_of_week':
			#df['day_of_week']=df.index.dayofweek
			df['day_of_week']=df["datetime"].dt.dayofweek
			df["day_of_week_norm"] = 2 * math.pi * df["day_of_week"] / 6.0 #df["day_of_week"].max() # We normalize x values to match with the 0-2π cycle
			df["cos_day_of_week"] = np.cos(df["day_of_week_norm"])
			df["sin_day_of_week"] = np.sin(df["day_of_week_norm"])
			if show_plot:
				df.plot.scatter(x="cos_day_of_week", y="sin_day_of_week").set_aspect('equal')
			df = df.drop(columns=['day_of_week_norm'])
		
		elif cyclical_feature == 'month_of_year':
			#df['month']=df.index.month
			df['month']=df["datetime"].dt.month
			df["month_norm"] = 2 * math.pi * df["month"] / 12.0 #df["day_of_week"].max() # We normalize x values to match with the 0-2π cycle
			df["cos_month"] = np.cos(df["month_norm"])
			df["sin_month"] = np.sin(df["month_norm"])
			if show_plot:
				df.plot.scatter(x="cos_month", y="sin_month").set_aspect('equal')
			df = df.drop(columns=['month_norm'])
		
		elif cyclical_feature == 'weekend':
			df["weekend"] = df["datetime"].dt.dayofweek >= 5
			df = df.astype({'weekend': 'float32'})
		
		else:
			print(f"{cyclical_feature} is an invalid cyclical feature!")
	
	if show_df:
		display(df)
	
	return df
