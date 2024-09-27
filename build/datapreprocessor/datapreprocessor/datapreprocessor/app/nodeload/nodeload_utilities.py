"""'
Created on Thursday Feb 16 15:00:00 2023
@author: Siby Plathottam
"""

import os
import calendar
import os
import pickle
import re
import itertools
import warnings
from typing import List, Set, Dict, Tuple, Optional, Union
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import default_rng
from pandas.tseries.offsets import DateOffset
from tqdm import tqdm
try:
	import seaborn as sns
except ImportError:
	warnings.warn('seaborn failed to import', ImportWarning)

from datapreprocessor.model.distmodel import DistModel
from datapreprocessor.app.nodeload.timeseries_data_utilities import get_time_series_dataframe,add_datetime,get_n_days_in_df

def find_df_timeperiod(df,datetime_column="datetime"):
	"""Upsample origina time series"""
	
	delta_t_seconds = (df[datetime_column].iloc[-1]-df[datetime_column].iloc[-2]).total_seconds()
	delta_t_minutes = int(delta_t_seconds/60.0)
	print(f"Original time period:{delta_t_seconds} seconds ({delta_t_minutes} minutes)")	

def get_upsampled_df(df_timeseries,upsample_time_period="15Min",datetime_column="datetime",index_name="block_index"):
	"""Upsample original time series"""
	print(f"Upsampling to {upsample_time_period}...")    
	find_df_timeperiod(df_timeseries,datetime_column=datetime_column)
	
    #print(f"Columns in upsampled df:{list(df_timeseries.columns)}")
	df_timeseries_upsampled = df_timeseries.set_index(datetime_column)

	df_timeseries_upsampled = df_timeseries_upsampled.resample(upsample_time_period,label='left')#convention='start')#closed="left")
	df_timeseries_upsampled = df_timeseries_upsampled.interpolate(method='linear')
	
	df_timeseries_upsampled = df_timeseries_upsampled.reset_index()
	if index_name:
		df_timeseries_upsampled.index.name = index_name
	
	delta_t_minutes = int((df_timeseries_upsampled[datetime_column].iloc[-1]-df_timeseries_upsampled[datetime_column].iloc[-2]).total_seconds()/60.0)
	print(f"Resampled time period:{delta_t_minutes} minutes")
	ts = df_timeseries_upsampled[datetime_column].iloc[-1] #pd.Timestamp('2017-01-01 09:10:11')	  
	df_timeseries_upsampled.loc[len(df_timeseries_upsampled),datetime_column] = ts + DateOffset(minutes=delta_t_minutes) #Adding an additional timestep at the end since resampling doesn't add it
	df_timeseries_upsampled = df_timeseries_upsampled.fillna(method="ffill")
	
	return df_timeseries_upsampled

def get_downsampled_df(df_timeseries,downsample_time_period="1S",datetime_column="datetime",index_name="block_index"):
	"""Downsample origina time series"""
	print(f"Downsampling to {downsample_time_period}...")
	find_df_timeperiod(df_timeseries,datetime_column=datetime_column)
	#print(f"Columns in upsampled df:{list(df_timeseries.columns)}")
	df_timeseries_downsampled = df_timeseries.set_index(datetime_column)
    
	df_timeseries_downsampled = df_timeseries_downsampled.resample(downsample_time_period).mean()  # '5T' represents 5-minute interval '1S' represents 1 second interval

	df_timeseries_downsampled = df_timeseries_downsampled.reset_index()
	if index_name:
		df_timeseries_downsampled.index.name = index_name
	print("After downsampling...")
	find_df_timeperiod(df_timeseries_downsampled,datetime_column=datetime_column)
	
	return df_timeseries_downsampled

def find_unique_loads(column_names):
    
    unique_loads = set()

    #pattern1 = r'customer_\d+_(\d+\.\d+)_' #Pattern for solar home data
    pattern1 = r'gross_load_customer_(\d+\.\d+)_\d+' #Pattern for solar home data
    pattern2 = r'gross_load_customer_(\d+\.\d+)' #Pattern for solar home data
    pattern3 = r'Load_\w+_'   #Pattern for time series data

    for col in column_names:
        match1 = re.search(pattern1, col)
        match2 = re.search(pattern2, col)
        match3 = re.search(pattern3, col)
        if match1:
            unique_loads.add(match1.group(1))
        elif match2:
            unique_loads.add(match2.group(1))
        elif match3:
            unique_loads.add(match3.group(0)[:-1])

    print(f"Found following {len(unique_loads)} unique_loads:")
    
    return unique_loads

def get_load_dict(df_timeseries,timeseries_file,load_identifier="load_"):
	"""Get load dict"""	
	
	load_type_col_names =[col_name for col_name in list(df_timeseries.keys()) if load_identifier in col_name.lower()]
	print(f"Found {len(load_type_col_names)} loads in {timeseries_file}")
	
	load_types = find_unique_loads(load_type_col_names)
	
	load_types = {load.replace('Load_', '') for load in load_types}
	load_dict ={}

	for load_type in load_types:
		if load_identifier == "load_":
			pattern = f'{load_type}_\d' #pattern for AMI load
		elif load_identifier == "customer_":
			pattern = f'customer_\d*_{load_type}_gross_load' #pattern for AMI load
		else:
			raise ValueError("Invalid load identifier")
			
		print(f"Filtering for load type pattern:{pattern}")
		#load_dict.update({load_type:[load_name for load_name in load_type_col_names if load_type in load_name]})
		load_dict.update({load_type:[name for name in load_type_col_names if re.search(pattern, name)]})

	print(f"Total identified loads:{sum(len(values) for values in load_dict.values())} in {timeseries_file}")
	
	return load_dict

def create_average_timeseries_profiles(timeseries_files,month,convert_to_kW=False,sample_time_period="15Min",datetime_column = "datetime",index_name="block_index"):
	"""Create load profiles by averaging across individual loads"""
	
	assert len(timeseries_files) == 1, f"Will only work with one timeseries file for now, but found:{timeseries_files}"
	df_averaged_load = pd.DataFrame()
	load_dict = {}
	
	for timeseries_file in timeseries_files:
		df_timeseries = get_time_series_dataframe(timeseries_file)
		load_dict = get_load_dict(df_timeseries,timeseries_file)
	
	load_dict = {load_type: value for load_type, value in load_dict.items() if value} #Check and remove empty lists
	load_types = list(load_dict.keys())
	print(f"Following {len(load_types)} load types were found:{load_types}")
	
	if datetime_column not in df_timeseries:
		print("'datetime' column not found in time series dataframe -- adding datetime column to time series dataframe")		
		df_timeseries = add_datetime(df_timeseries,show_details = True)
	else:
		print(f"{datetime_column} column found in time series dataframe -- using datetime column from time series dataframe")
		assert not any('24:' in time_str for time_str in df_timeseries[datetime_column].to_list()), "timestamp contain 24 which is not valid" 
	df_averaged_load[datetime_column]  = pd.to_datetime(df_timeseries[datetime_column], infer_datetime_format=True) #, format='%Y-%m-%d-%H:%M:%S'
	
	assert df_averaged_load[datetime_column].dtype == 'datetime64[ns]', "Should be date time data type"
	
	df_averaged_load[datetime_column]  = df_averaged_load[datetime_column].dt.strftime('%Y-%m-%d-%H:%M:%S')
	df_averaged_load[datetime_column]  = pd.to_datetime(df_averaged_load[datetime_column], format='%Y-%m-%d-%H:%M:%S')
	
	df_averaged_load = df_averaged_load[df_averaged_load[datetime_column].dt.month == month]
	found_month = list(set(df_averaged_load[datetime_column].dt.month.to_list()))[0]
	found_year = list(set(df_averaged_load[datetime_column].dt.year.to_list()))[0]
	days_in_month1 = calendar.monthrange(int(found_year), int(found_month))[1]
	days_in_month = get_n_days_in_df(df_averaged_load,datetime_column = datetime_column)
	print(f"Expected days in month:{found_month}:{days_in_month1}, Found days:{days_in_month}")
	
	for load_type in load_dict.keys():
		print(f"Averaging {len(load_dict[load_type])} loads of {load_type}")
		df_averaged_load[load_type] = df_timeseries[load_dict[load_type]].mean(axis=1)
	load_type_example = load_types[0]
	total_kWh_before_resampling = sum(df_averaged_load[load_type_example].values) #Find total kWh before resampling for verification
	
	del_hour = 24/len(df_averaged_load[df_averaged_load[datetime_column].dt.day==1]) #Find the time delta between each load measurement in hours
	if convert_to_kW:
		print(f"Converting kWh to kW with a time delta of:{del_hour} hour")
		df_averaged_load = convert_kwh_columns(df_averaged_load,list(load_dict.keys()),del_hour)

	df_averaged_load = get_updownsampled_df(df_averaged_load,datetime_column=datetime_column,sample_time_period=sample_time_period,index_name=index_name)
	
	del_hour = 24/len(df_averaged_load[df_averaged_load[datetime_column].dt.day==1]) #Find the time delta between each load measurement in hours
	total_kWh_after_resampling = sum(df_averaged_load[load_type_example].values*del_hour) #Find total kWh after resampling for verification
	print(f"Load type:{load_type_example} - Total kWH:Before resampling:{total_kWh_before_resampling:.2f},After resampling with time delta of {del_hour} hr:{total_kWh_after_resampling:.2f}")
	
	df_averaged_load["day_of_week"] = df_averaged_load[datetime_column].dt.weekday
	df_averaged_load["weekend"] = df_averaged_load["day_of_week"] >= 5
		
	assert load_types == [load_type for load_type in list(df_averaged_load.columns) if load_type not in ['day_of_week', 'weekend',datetime_column]], "Expected load types in dictionary and data frame to be same"
			
	n_timesteps_per_day = len(df_averaged_load[df_averaged_load[datetime_column].dt.day==1]) #Find number of time stamps in one day
	df_averaged_day_load = pd.DataFrame()
	df_averaged_day_load[datetime_column] = df_averaged_load.loc[0:n_timesteps_per_day-1,datetime_column]
	
	for j,load_type in enumerate(load_types):
		cols = []
		for i in range(0,days_in_month):
			cols.append(f"{load_type}_day_{i+1}")			
			df_averaged_day_load[f"{load_type}_day_{i+1}"] = df_averaged_load.loc[i*n_timesteps_per_day:(i+1)*n_timesteps_per_day-1,load_type].values
		df_averaged_day_load.insert(j+1,load_type,df_averaged_day_load[cols].mean(axis=1).values)
		#df_averaged_day_load = df_averaged_day_load.drop(columns=cols)
	
	print(f"Expected length of averaged load df:{days_in_month*n_timesteps_per_day} - actual:{len(df_averaged_load)}") #Days in a month*n_timesteps_per_day
	df_averaged_load_weekend = df_averaged_load[df_averaged_load["weekend"]==True]
	print(f"Length of weekend averaged load df:{len(df_averaged_load_weekend)}")
	df_averaged_load_weekday = df_averaged_load[df_averaged_load["weekend"]==False]
	print(f"Length of weekday averaged load df:{len(df_averaged_load_weekday)}")
	
	weekend_days = int(len(df_averaged_load_weekend)/n_timesteps_per_day)
	weekday_days = int(len(df_averaged_load_weekday)/n_timesteps_per_day)
	
	print(f"Total weekend days:{weekend_days}")
	print(f"Total weekday days:{weekday_days}")
	
	df_averaged_load_weekend.reset_index(drop=True, inplace=True)
	df_averaged_load_weekend.index.rename(index_name, inplace=True)
	df_averaged_load_weekday.reset_index(drop=True, inplace=True)
	df_averaged_load_weekday.index.rename(index_name, inplace=True)	 
	
	for j,load_type in enumerate(load_types):
		weekend_cols = []
		weekday_cols = []
		for i in range(0,weekend_days):
			weekend_cols.append(f"{load_type}_weekend_{i+1}")
			df_averaged_day_load[f"{load_type}_weekend_{i+1}"] = df_averaged_load_weekend.loc[i*n_timesteps_per_day:(i+1)*n_timesteps_per_day-1,load_type].values
		for i in range(0,weekday_days):
			weekday_cols.append(f"{load_type}_weekday_{i+1}")
			df_averaged_day_load[f"{load_type}_weekday_{i+1}"] = df_averaged_load_weekday.loc[i*n_timesteps_per_day:(i+1)*n_timesteps_per_day-1,load_type].values
			
		df_averaged_day_load.insert(j+1,f"{load_type}_weekend",df_averaged_day_load[weekend_cols].mean(axis=1).values)
		df_averaged_day_load.insert(j+1,f"{load_type}_weekday",df_averaged_day_load[weekday_cols].mean(axis=1).values)
		#df_averaged_day_load = df_averaged_day_load.drop(columns=weekend_cols)
		#df_averaged_day_load = df_averaged_day_load.drop(columns=weekday_cols)
	
	#display(df_averaged_day_load.head())
	
	for load_type in load_types:
		print(f"Difference in means for {load_type}:{df_averaged_load[load_type].mean()-(df_averaged_day_load[f'{load_type}_weekday'].mean()*weekday_days+df_averaged_day_load[f'{load_type}_weekend'].mean()*weekend_days)/(weekday_days+weekend_days):.4f}")
		
	return df_averaged_load,df_averaged_day_load
	
def plot_averaged_load(df,days=[1],selected_load_types=[],datetime_column = "datetime"):
	"Plot average load across different instances of same load type"
	
	print(f"Found load types:{selected_load_types}")
	
	for load_type in selected_load_types:
		for day in days:
			df[df[datetime_column].dt.day==day].plot(x=datetime_column,y=[load_type])

def plot_averaged_day_load(df,selected_load_types=[],datetime_column = "datetime"):
	"Plot average load across different instances and days of same load type"
		
	print(f"Found load types:{selected_load_types}")
	
	for load_type in selected_load_types:
		df.plot(x=datetime_column,y=[load_type,f"{load_type}_weekday",f"{load_type}_weekend"])

def get_df_load_fraction(df_averaged_day_load,df_load_fraction,month,datetime_column = "datetime"):	  
	df_node_load = pd.DataFrame()
	df_node_load[datetime_column] = df_averaged_day_load[datetime_column]
	node_load_fraction_dict = df_load_fraction.iloc[0,:].to_dict()
	df_node_load[f'node_load_equal'] = df_averaged_day_load.apply(lambda row: mean([row[load_type]*1.0 for load_type in list(node_load_fraction_dict.keys())]), axis=1) #Each load with equal fraction
	n_samples = len(df_load_fraction)
	for i in tqdm(range(n_samples)):
		node_load_fraction_dict = df_load_fraction.iloc[i,:].to_dict()
		df_node_load[f'node_load_{i}'] = df_averaged_day_load.apply(lambda row: sum([row[load_type]*node_load_fraction_dict[load_type] for load_type in list(node_load_fraction_dict.keys())]), axis=1)
		df_node_load[f'node_load_weekend_{i}'] = df_averaged_day_load.apply(lambda row: sum([row[load_type+"_weekend"]*node_load_fraction_dict[load_type] for load_type in list(node_load_fraction_dict.keys())]), axis=1)
		df_node_load[f'node_load_weekday_{i}'] = df_averaged_day_load.apply(lambda row: sum([row[load_type+"_weekday"]*node_load_fraction_dict[load_type] for load_type in list(node_load_fraction_dict.keys())]), axis=1)
	
	df_node_load.to_csv("df_node_load")
	
	node_load_dict = {"df_node_load":df_node_load,"df_load_fraction":df_load_fraction}
	
	print(f"Saving node loads in: m-{month}_node_load.pickle")
	with open(f'm-{month}_node_load.pickle', 'wb') as file:
		pickle.dump(node_load_dict, file, protocol=pickle.HIGHEST_PROTOCOL) # save dictionary to pickle file
	
	return df_node_load

def get_load_type_fraction(available_load_types,min_n_load_types = 2,n_samples = 2):
	"""Create load faction"""
	rng = default_rng()
	
	if "misc" in available_load_types:
		available_load_types.remove("misc")
	print(f"Available {len(available_load_types)} loads:{available_load_types}")
	node_load_fraction_dict = {}
	for i in range(n_samples):
		n_load_types = rng.integers(min_n_load_types,len(available_load_types)+1)
		#print(f"Node will contain {n_load_types} loads")
		load_types_at_node = list(rng.choice(available_load_types,size = n_load_types,replace=False))
		#print(f"Load types at node: {load_types_at_node}")
		fractions_for_load_type = rng.dirichlet(np.ones(len(load_types_at_node)),size=1) #In Dirichelt distribution fractions will sum to one
		fractions_for_load_type = fractions_for_load_type.tolist()[0]
		#print(f"fractions_per_load_type: {fractions_for_load_type}")

		node_load_fraction_dict.update({i:{load_type:fraction_per_load_type for load_type,fraction_per_load_type in zip(load_types_at_node,fractions_for_load_type)}})
		node_load_zero_fraction_dict = {load_type:0.0 for load_type in available_load_types if load_type not in load_types_at_node}
		node_load_fraction_dict[i].update(node_load_zero_fraction_dict)
		#print(f"Sum of fractions:{sum(node_load_fraction_dict[i].values())}")
	df_load_fraction = pd.DataFrame.from_dict(node_load_fraction_dict, orient='index')
	
	return df_load_fraction

def get_df_node_load_fraction(pickle_file):
	# laod a pickle file
	with open(pickle_file, "rb") as file:
		loaded_dict = pickle.load(file)
		
	return loaded_dict["df_node_load"],loaded_dict["df_load_fraction"]

def get_scaling_factor(Pnominal,df_node_load,df_load_fraction):
	"Calculate scaling factor for each distribution system node"
	
	nodes = list(Pnominal.keys()) #Get list of nodes
	df_scaling_factor = pd.DataFrame(columns = nodes)
	
	for sample_id in tqdm(range(len(df_load_fraction))): #Loop through each load fraction sample
		for node_id in nodes:			 
			df_scaling_factor.loc[sample_id,node_id] = Pnominal[node_id]/max(df_node_load[f"node_load_{sample_id}"])  #Calculate scaling factor by dividing the nominal node load with the maximum load in the time series	  
	
	return df_scaling_factor

def get_rated_node_loads(case_file,n_nodes=-1):
	"""Return P and Q in kW and kVAR respectively"""
	
	dss=DistModel()
	print(f"Loading OpenDSS case file:{case_file}")
	dss.load_case(case_file)
	print(f"Found {len(dss.S0['P'].keys())} nodes")
	count_phases(dss.S0['P'])
	Pnominal = dss.S0['P']
	Qnominal = dss.S0['Q']
	
	if n_nodes == -1 or n_nodes>=len(Pnominal):
		print(f"Selecting all {len(Pnominal)} nodes...")
	else:
		print(f"Selecting first {n_nodes} nodes from {len(Pnominal)} nodes")
		Pnominal = dict(itertools.islice(Pnominal.items(), n_nodes))
		Qnominal = {k: v for k, v in Qnominal.items() if k in Pnominal}
	
	return Pnominal,Qnominal

def count_phases(load_dict):
	"""
	Count the number of occurrences of each phase in a dictionary.
	
	Args:
	- dictionary: a dictionary containing the loads at each node
	
	Returns: dictionary
	
	Prints and returns the number of occurrences of each phase (a, b, and c).
	"""
	# Define regular expression patterns to match strings that end with 'a', 'b', or 'c' (followed by any other character)
	pattern_a = re.compile(r'a.$')
	pattern_b = re.compile(r'b.$')
	pattern_c = re.compile(r'c.$')

	# Initialize counter variables
	phase_a_count = 0
	phase_b_count = 0
	phase_c_count = 0
	
	# Iterate through the dictionary of strings and check if the patterns match
	for key in load_dict:
		
		if pattern_a.search(key) or key.endswith('a') or key.endswith('_1'):
			phase_a_count += 1
		elif pattern_b.search(key) or key.endswith('b') or key.endswith('_2'):
			phase_b_count += 1
		elif pattern_c.search(key) or key.endswith('c') or key.endswith('_3'):
			phase_c_count += 1
	print(f"Number of phase_a: {phase_a_count}")
	print(f"Number of phase_b: {phase_b_count}")
	print(f"Number of phase_c: {phase_c_count}")
	# Return a dictionary with the counts of strings that match the patterns
	return {'phase_a': phase_a_count, 'phase_b': phase_b_count, 'phase_c': phase_c_count}

def generate_load_array(df_averaged_day_load,available_load_types,min_n_load_types=2,show_details=False):
	"""Select n load types and return an array of base load profiles"""
	
	rng = default_rng()
	n_load_types_to_be_selected = rng.integers(min_n_load_types,len(available_load_types)+1)
	selected_load_types = list(rng.choice(available_load_types,size = n_load_types_to_be_selected,replace=False))
	if show_details:
		print(f"Selecting following {n_load_types_to_be_selected} load types:{selected_load_types}")
	
	return {'load_array':df_averaged_day_load[[f"{load_type}" for load_type in selected_load_types]].values,"load_types":selected_load_types}

def match_avg_energy(L,E_kwh=50,show_details=False): #,nProfile=7,nPts=96
	"""Generate profiles by taking random fractions from existing profiles"""
	
	rng = default_rng()
	#L=np.random.rand(nPts,nProfile) #Replaced by averaged load profile
	nPts= L.shape[0] #96 #Number of time steps
	nProfile= L.shape[1] #7	   #Number of base profiles of the load types
	
	if show_details:
		print(f"Number of base profiles:{nProfile},Number of time steps:{nPts}")
		print(f"Base profile {len(L.mean(0))} means:{L.mean(0)}")
		print(f"Base load:{E_kwh}")
	
	L=L/L.mean(0)  #Divide individual profiles with their means to normalize the profiles
	
	#w=np.random.rand(nProfile)	 #Replaced by Dirichlet distribution
	#w=w/sum(w)					 #Replaced by Dirichlet distribution
	w = rng.dirichlet(np.ones(nProfile),size=1)[0] #In Dirichlet distribution fractions will sum to one
	x=L*w #Multiply base load profile with weights to get weighted load profile
	x=x.sum(1) #Sum weighted loads at each time step
	
	assert sum(x)/nPts>=1-1e-8 and sum(x)/nPts<=1+1e-8 # avg unscaled energy will be 1 per dt, so over an hr it will be 1 kwh
	load_shape=E_kwh*x #Get node load profile
	
	#plt.plot(load_shape)
	return load_shape

def scale_weighted_load(L,k_scale,show_details=False): #,nProfile=7,nPts=96
	"""Generate profiles by taking random fractions from existing profiles"""
	
	rng = default_rng()	   
	nPts= L.shape[0] #96 #Number of time steps
	nProfile= L.shape[1] #7	   #Number of base profiles of the load types
	
	assert len(k_scale) == nProfile, "Each base profile has a seperate scaling factor"
	
	if show_details:
		print(f"Number of base profiles:{nProfile},Number of time steps:{nPts}")
		print(f"Base profile {len(L.mean(0))} means:{L.mean(0)}")
		print(f"Scaling factor:{k_scale}")
		
	w = rng.dirichlet(np.ones(nProfile),size=1)[0] #In Dirichlet distribution fractions will sum to one
	
	x=L*w #Multiply base load profile with weights to get weighted load profile
	x=k_scale*x #Multiply weighted load profile with scaling factor for each node
	load_shape=x.sum(1) #Sum weighted loads at each time step to get node load profile
	#assert load_shape.max()< Pnominal, "Profile max should be less thatn nominal"
	#print(f"mean:{load_shape.mean()},Max:{load_shape.max()}")
	
	return load_shape

def generate_load_node_profiles(df_averaged_day_load,case_file,n_nodes=-1,n_days=1,start_year = 2016,start_month=1,start_day=1,scaling_type="simple",datetime_column = "datetime"):
	"""Generate profiles for each node in openDSS distribution feeder using the averaged day load"""
	
	available_load_types = [load_type for load_type in list(df_averaged_day_load.columns) if load_type not in ['day_of_week', 'weekend',datetime_column]] #Remove non-load columns	
	available_load_types = [load_type for load_type in available_load_types if not any(s in load_type for s in ["_day_","_weekday","_weekend"])] #Remove unnecssary loads
	#missing_load_types = [load_type for load_type in available_load_types if load_type not in list(df_averaged_day_load.columns.unique())]
	#if missing_load_types:
	#	print(f"Following base profiles not found:{missing_load_types}... removing")
	#available_load_types = [load_type for load_type in available_load_types if load_type not in missing_load_types]
	
	print(f"Creating energy profile for year:{start_year},month:{start_month} with averaged profiles of following {len(available_load_types)} loads:{available_load_types}")
	Pnominal,Qnominal = get_rated_node_loads(case_file,n_nodes) #Get nominal spot loads at each node	
	df_node_load = pd.DataFrame()
	time_stamps = []
	load_profiles = {node:[] for node in Pnominal.keys()}
	n_timesteps_per_day = len(df_averaged_day_load)
	print(f"Number of time steps per day:{n_timesteps_per_day}")
	time_interval_in_minutes = f"{int((24*60)/n_timesteps_per_day)} min" #"30min" #Calculating time interval in minutes
	print(f"Expected time interval in minutes:{time_interval_in_minutes}")	
	time_interval_in_minutes = f"{int((df_averaged_day_load[datetime_column].iloc[-1]-df_averaged_day_load[datetime_column].iloc[-2]).total_seconds()/60.0)} min"	
	print(f"Calculated time interval in minutes:{time_interval_in_minutes}")

	load_node_dict = {node:generate_load_array(df_averaged_day_load,available_load_types,min_n_load_types=2) for node in Pnominal.keys()} #Assign base load type profiles to each node
	
	for i in tqdm(range(n_days)): #Each sample is a new day. The base load profiles are fixed for each node for all the samples
		month = str(start_month).zfill(2)
		day = str(start_day).zfill(2)
		start_date= f"{start_year}-{month}-{day}" # "2016-02-01"
		timestamp_range = pd.date_range(start=start_date,periods=n_timesteps_per_day,freq=time_interval_in_minutes)# end=" 2016-02-01" #Create time stamps
		month = list(set(timestamp_range.month.to_list()))[0]
		time_stamps.extend(timestamp_range.to_list())
		for node in Pnominal.keys(): #node = 's1a'
			load_node_dict[node]["Pnominal"] =	 Pnominal[node]
			if scaling_type == "simple":
				E_kwh = Pnominal[node]
				L_profile_node = match_avg_energy(load_node_dict[node]["load_array"],E_kwh)
			elif scaling_type == "multi":
				if "k_scale" not in load_node_dict[node].keys(): #Calculate scaling factor one for each node
					#load_node_dict[node]["k_scale"] = Pnominal[node]/load_node_dict[node]["load_array"].max(axis = 0)	#Calculate scaling factor by dividing the nominal node load with the maximum load in the time series
					load_node_dict[node]["k_scale"] = Pnominal[node]/load_node_dict[node]["load_array"].mean(axis = 0)	#Calculate scaling factor by dividing the nominal node load with the maximum load in the time series
				L_profile_node = scale_weighted_load(load_node_dict[node]["load_array"],load_node_dict[node]["k_scale"]) #Find weighted scaled node load profle
			else:
				raise ValueError(f"Invalid scaling type:{scaling_type}")
			load_profiles[node].extend(L_profile_node)
		
		start_day = start_day + 1 #Increment day
		#assert start_day <= calendar.monthrange(start_year, start_month)[1], f"Day of month {start_day} is greater than number of days in the month:{calendar.monthrange(start_year, start_month)[1]}"
		
		if start_day > calendar.monthrange(start_year, start_month)[1]:
			print(f"Reseting since day count {start_day} is greater than:{calendar.monthrange(start_year, start_month)[1]}")
			start_day = 1
	
	df_node_load = pd.DataFrame.from_dict(load_profiles) #Faster
	df_node_load.insert(0,datetime_column,time_stamps) #Insert at first columns
	
	return df_node_load,load_node_dict

def generate_node_voltage_profiles(df_voltage,case_file,n_nodes=-1,n_days=1,start_year = 2016,start_month=1,start_day=1,scaling_type="simple"):
	"""Generate profiles for each node in openDSS distribution feeder using the averaged day load"""
		
	print(f"Creating node voltage profiles for year:{start_year},month:{start_month}")
	Pnominal,Qnominal = get_rated_node_loads(case_file,n_nodes) #Get nominal spot loads at each node
	df_node_load = pd.DataFrame()
	time_stamps = []
	voltage_profiles = {node:[] for node in Pnominal.keys()}
	n_timesteps_per_day = len(df_voltage)
	print(f"Number of time steps per day:{n_timesteps_per_day}")
	time_interval_in_minutes = f"{int((24*60)/n_timesteps_per_day)} min" #"30min" #Calculating time interval in minutes
	print(f"Time interval in minutes:{time_interval_in_minutes}")
	time_interval_in_minutes = f"{int((df_voltage['datetime'].iloc[-1]-df_voltage['datetime'].iloc[-2]).total_seconds()/60.0)} min"
	
	print(f"Time interval in minutes:{time_interval_in_minutes}")
	node_voltage_dict = {node:df_voltage["vmag_pmu_28_1"].values for node in Pnominal.keys()} #Assign base load type profiles to each node
	
	for i in tqdm(range(n_days)): #Each sample is a new day. The base load profiles are fixed for each node for all the samples
		month = str(start_month).zfill(2)
		day = str(start_day).zfill(2)
		start_date= f"{start_year}-{month}-{day}" # "2016-02-01"
		timestamp_range = pd.date_range(start=start_date,periods=n_timesteps_per_day,freq=time_interval_in_minutes)# end=" 2016-02-01" #Create time stamps
		month = list(set(timestamp_range.month.to_list()))[0]
		time_stamps.extend(timestamp_range.to_list())
				
		start_day = start_day + 1 #Increment day
		
		if start_day > calendar.monthrange(start_year, start_month)[1]:
			print(f"Reseting since day count {start_day} is greater than:{calendar.monthrange(start_year, start_month)[1]}")
			start_day = 1
	
	df_node_voltage = pd.DataFrame.from_dict(node_voltage_dict) #Faster
	df_node_voltage.insert(0,'datetime',time_stamps) #Insert at first columns
	
	return df_node_voltage,node_voltage_dict

def generate_voltage_array(df_voltage,available_pmu,show_details=False):
	"""Select n load types and return an array of base load profiles"""
	
	rng = default_rng()
	n_load_types_to_be_selected = rng.integers(min_n_load_types,len(available_load_types)+1)
	selected_load_types = list(rng.choice(available_load_types,size = n_load_types_to_be_selected,replace=False))
	if show_details:
		print(f"Selecting following {n_load_types_to_be_selected} load types:{selected_load_types}")
	
	return {'load_array':df_averaged_day_load[[f"{load_type}" for load_type in selected_load_types]].values,"load_types":selected_load_types}

def match_avg_energy_original(nProfile=7,nPts=96,E_kwh=50):
	L=np.random.rand(nPts,nProfile)
	L=L/L.mean(0)
	w=np.random.rand(nProfile)
	#print(w)
	w=w/sum(w)
	#print(w)
	x=L*w
	x=x.sum(1)
	print(sum(x)/nPts)
	assert sum(x)/nPts>=1-1e-8 and sum(x)/nPts<=1+1e-8 # avg unscaled energy will be 1 per dt, so over an hr it will be 1 kwh
	load_shape=E_kwh*x

	return load_shape

def convert_kwh_to_kw(kWh_values,del_hour):
	kW_values = kWh_values/del_hour
	
	return kW_values

def convert_kwh_columns(df,columns,del_hour):
	"""Convert kWh unit to kW units"""
	
	print(f"Converting following columns:{columns} to kWh")
	for column in columns:
		df[column] = convert_kwh_to_kw(df[column].values,del_hour)
		
	return df

def plot_distribution(df,column_names: List[str]=[],column_index_names:Union[str, List[str]]=[],plot_type: str='histogram',
					  max_value:float=100000.0,bins:int=100,fig_size=(10, 10),plot_folder='',plot_title='',show_plot=True,show_means=True,font_size=16,line_styles={},kde_kws = {'linewidth': 3,'bw_adjust':6},y_label = "Value"):
	"""Plot histogram with important statistics"""
	sns.set(font_scale = 2)
	print(f"Number of elements in dataframe:{len(df)}")
	
	if plot_type == 'histogram':
		n_samples = len(df)
		plt.figure(figsize=fig_size)
		ax=df[column_names].plot(kind='hist',histtype='step',bins=bins,figsize=fig_size,alpha=0.5)
		if show_means:			  
			for i,column_name in enumerate(column_names):
				plt.axvline(df[column_name].mean(),c='C'+str(i),linestyle='-')
				plt.axvline(df[column_name].median(),c='C'+str(i),linestyle='--')
				plt.axvline(df[column_name].quantile(0.25),c='C'+str(i),linestyle=':')
				plt.axvline(df[column_name].quantile(0.75),c='C'+str(i),linestyle=':')
		plt.xlabel('Value')
		plt.title(f"{plot_title}-{n_samples} samples")
	
	elif plot_type == 'box':
		n_samples = len(df)
		plt.figure(figsize=fig_size)
		medians = df[column_names].median(axis=0)
		means = df[column_names].mean(axis=0)
		std_devs = df[column_names].std(axis=0)
		ax,lines = df[column_names].boxplot(showmeans = True,return_type='both')
				
		for i, line in enumerate(lines['medians']):
			x, y = line.get_xydata()[1]
			text = 'M={:.2f}\n μ={:.2f}\n σ={:.2f}'.format(medians[i],means[i], std_devs[i])
			ax.annotate(text, xy=(x, y))
		plt.title(f"{plot_title}-{n_samples} samples")
		plt.ylabel(y_label)
	
	elif plot_type == 'bar':
		plt.figure(figsize=fig_size)
		ax = df.loc[index_names,column_names].plot.bar()
	
	elif plot_type == 'density_pandas':
		plt.figure(figsize=fig_size)
		ax = df[column_names].plot.kde(figsize=fig_size)
	
	elif plot_type == 'density':
		plt.figure(figsize=fig_size)
		n_samples = len(df)
		for column_name in column_names:
			# Draw the density plot
			if column_name in line_styles:
				line_style = line_styles[column_name]
				print(f'Changing line style for {column_name} to {line_style}')
			else:
				line_style = '-'
			kde_kws.update({'linestyle':line_style})
			
			ax = sns.distplot(df[column_name], hist = False, kde = True,
						 kde_kws =kde_kws,
						 label = column_name)
	
			# Plot formatting
		plt.legend(prop={'size': 20}, title = "Column name")
		plt.title(f"{plot_title}-{n_samples} samples")
		
		plt.ylabel('Density')

	elif plot_type == 'violin':
		plt.figure(figsize=fig_size)
		df_melted = df[column_names].melt(var_name="Feature", value_name="Value")
		sns.set_context("paper")
		sns.set_theme(style="whitegrid")
		ax = sns.violinplot(x="Feature", y="Value",data=df_melted[df_melted["Value"]<max_value],  inner="quartile",scale='width')
		
		plot_title = plot_title + f'(max <= {max_value})'
		
	#ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
	#ax.set_title(plot_title, fontsize=font_size)  # or size, alternatively
	
	if not os.path.exists(plot_folder):
		print(f"Creating folder:{plot_folder}...")
		os.mkdir(plot_folder)
	
	plot_file_name = os.path.join(plot_folder,plot_title)
	plot = ax.get_figure()
	print(f"Saving figure in {plot_file_name}....")
	plot.savefig(plot_file_name, dpi=400, bbox_inches = "tight")
	
	return ax

def check_and_create_folder(folder_path):
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)
		print(f'Folder "{folder_path}" created successfully.')

def get_updownsampled_df(df:pd.DataFrame,datetime_column:str,sample_time_period:str,index_name:str=""):

	current_timedelta = df[datetime_column].diff().min()	
	#current_timedelta = pd.to_timedelta(current_timedelta, unit='s')
	desired_timedelta = pd.to_timedelta(sample_time_period)
	print(f"Current time delta is:{current_timedelta}, desired time delta is:{desired_timedelta}")
	if desired_timedelta > current_timedelta: # Downsample
		df = get_downsampled_df(df,datetime_column=datetime_column,downsample_time_period=sample_time_period,index_name=index_name)
	elif desired_timedelta < current_timedelta: # Upsample
		df = get_upsampled_df(df,datetime_column=datetime_column,upsample_time_period=sample_time_period,index_name=index_name)
	else:
		print("No resampling required!")
	
	return df
