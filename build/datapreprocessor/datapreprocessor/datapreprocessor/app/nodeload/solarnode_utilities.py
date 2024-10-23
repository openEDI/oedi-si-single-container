"""'
Created on Wed November 30 10:00:00 2022
@author: Siby Plathottam
"""

from typing import List, Set, Dict, Tuple, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.random import default_rng

from datapreprocessor.app.nodeload.nodeload_utilities import get_updownsampled_df,get_rated_node_loads,find_unique_loads

def convert_solardata_to_timeseries(df:pd.DataFrame,n_days:int,n_customers:int = 10,show_details:bool=False,rename_dict:Dict = {},datetime_column="datetime"):
	"""Function to convert raw solar home data from Australia solar home data to time series data."""
	
	time_intervals = list(filter(lambda k: ':' in k, list(df.columns)))
	time_intervals.insert(0,time_intervals.pop())
	shifted_time_intervals = {old_time_interval:time_intervals[i] for i,old_time_interval in enumerate(list(filter(lambda k: ':' in k, list(df.columns))))}
	df = df.rename(columns =shifted_time_intervals)
	time_stamps = list(shifted_time_intervals.values())
	df_timeseries =pd.DataFrame()
	generator_capacities = []
	
	customer_ids = df["Customer"].unique()
	customer_ids = customer_ids[:n_customers] #Select first n customers
	print(f"Selecting {len(customer_ids)} customer ids:{customer_ids}")
	for customer_id in tqdm(customer_ids,position=0): #Loop through customer ids
		df_customer = df[df["Customer"]==customer_id]
		#print('Customer name:{}'.format(customer))
		for consumption_category in ["GG","GC"]: #loop through consumption categories
			df_customer_category = df_customer[df_customer['Consumption Category']==consumption_category].reset_index()
			date_index = 0
			date_time_index = 0
			assert len(df_customer_category['Generator Capacity'].unique()) == 1, "Only expects one generator capacity"
			generator_capacity = df_customer_category.loc[date_index,'Generator Capacity']
			if consumption_category == "GG": #Count generator capacities only once
				generator_capacities.append(generator_capacity)
			if consumption_category in rename_dict:
				consumption_category = rename_dict[consumption_category]
			customer_name = f"{consumption_category}_customer_{generator_capacity}_{customer_id}"
			
			for day in range(0,n_days): #loop through days
				if date_index in df_customer_category.index:
					date =	df_customer_category.loc[date_index,'date']
					for time_block in time_stamps: #loop through each time step in a day
						hour = time_block.split(":")[0].zfill(2)
						minutes = time_block.split(":")[1]
						seconds = "00"
						df_timeseries.loc[date_time_index,'date_block'] = date #pd.to_datetime(df.loc[date,'date'])
						df_timeseries.loc[date_time_index,'time_block'] = hour+":"+minutes+":"+seconds #time_block
						df_timeseries.loc[date_time_index,customer_name] = df_customer_category.loc[date_index, time_block]
						date_time_index = date_time_index +1
				else:
					if show_details:
						print(f"Skipping following missing date index {date_index}:{customer_id},{consumption_category}")
				date_index = date_index+1
	
	print(f"Total unique dates:{len(df_timeseries.date_block.unique())}")
	print(f"Unique {len(generator_capacities)} generator capacities:{generator_capacities}")
	df_timeseries.insert(0, datetime_column, df_timeseries['date_block'] + "-" + df_timeseries['time_block']) #Insert date time block
	df_timeseries[datetime_column]  = pd.to_datetime(df_timeseries[datetime_column], format='%d/%m/%Y-%H:%M:%S')
	
	if show_details == True:
		print(df_timeseries.head())
	
	return df_timeseries




def generate_solar_array(df_timeseries:pd.DataFrame,available_solar_types,min_n_solar_types:int=2,max_n_solar_types:int=5,show_details:bool=False,solar_column_identifier ="solar_power_customer",load_column_identifier ="gross_load_customer",datetime_column="datetime"):
	"""Select n load types and return an array of base load profiles"""
	
	rng = default_rng()
	n_solar_types_to_be_selected = rng.integers(min_n_solar_types,max_n_solar_types+1)

	selected_solar_types = list(rng.choice(available_solar_types,size = n_solar_types_to_be_selected,replace=False))
	if show_details:
		print(f"Selecting following {n_solar_types_to_be_selected} solar types:{selected_solar_types}")
	
	df_solar = pd.DataFrame()
	df_solar[datetime_column] = df_timeseries[datetime_column]

	#df_solar = pd.concat([df_timeseries.loc[:, df_timeseries.columns.str.endswith(f'_{capacity}_solar_power')].sample(n=1,axis='columns') for capacity in selected_solar_types],axis =1) #combine solar power from selected customers
	df_solar = pd.concat([df_timeseries.loc[:, df_timeseries.columns.str.startswith(f'{solar_column_identifier}_{capacity}_')].sample(n=1,axis='columns') for capacity in selected_solar_types],axis =1) #combine solar power from selected customers
	gross_loads = [solar.replace(f"{solar_column_identifier}",f"{load_column_identifier}") for solar in list(df_solar.columns)] #Get gross loads correponding to selected solar customers

	df_solar[gross_loads] = df_timeseries.loc[:,gross_loads].values
	df_solar["total_solar_power"] = df_solar.loc[:, df_solar.columns.str.contains(f'{solar_column_identifier}')].sum(axis=1) #df_node_solar.sum(axis=1)
	df_solar["total_gross_load"] = df_solar.loc[:, df_solar.columns.str.contains(f'{load_column_identifier}')].sum(axis=1) #df_node_gross_load.sum(axis=1)

	if show_details:
		print(df_solar.head())
		df_solar.plot(y="total_solar_power")
		df_solar.plot(y="total_gross_load")

	return df_solar,{"solar_types":selected_solar_types}

def generate_solar_node_profiles(df_timeseries:pd.DataFrame,casefile:str,selected_months:List = [1],n_solar_nodes:int=10,max_solar_penetration:float = 0.4,datetime_column:str="datetime",sample_time_period:str="15Min",index_name:str=""):
	"""Generate solar power and gross load profiles scaled to the distribution system node"""
	rng = default_rng()
	print(f"Creating solar node profiles for {casefile}")
	P,_ = get_rated_node_loads(case_file=casefile,n_nodes=n_solar_nodes)
	
	if n_solar_nodes<0: # -ve number implies n_solar_nodes=number of nodes
		n_solar_nodes=len(P) # add all nodes 

	assert n_solar_nodes <= len(P), f"THe number of solar nodes for which profiles are generated:{n_solar_nodes} should be less than or equal to the number of load nodes:{len(P)}!"

	solar_penetration = rng.uniform(low=0.01, high=max_solar_penetration,size =n_solar_nodes)
	
	df_timeseries = get_updownsampled_df(df_timeseries,datetime_column=datetime_column,sample_time_period=sample_time_period,index_name=index_name)
		
	df_timeseries = df_timeseries[df_timeseries['datetime'].dt.month.isin(selected_months)] #Filter out data for selected months
	print(f"Filtering for month:{selected_months} to get df of size:{len(df_timeseries)}")

	solar_nodes = list(rng.choice(list(P.keys()),size = n_solar_nodes,replace=False)) #Select a sub-set of the distribution system nodes as solar nodes
	solar_node_dict = {solar_node:{"solar_penetration":round(solar_penetration[i],3)} for i,solar_node in enumerate(solar_nodes)}
	
	available_solar_capacities = list(find_unique_loads(list(df_timeseries.columns)))
	print(f"Found {len(available_solar_capacities)} solar capacities")
	assert len(available_solar_capacities) >= 1, "Expected atleast 1 capacity"

	res={'datetime':df_timeseries["datetime"]}
	colChoice=list(df_timeseries.columns[df_timeseries.columns.str.startswith('solar')])
	for node in tqdm(solar_node_dict.keys()):
		np.random.shuffle(colChoice)
		if len(available_solar_capacities) > 1:
			selectedCols=colChoice[0:np.random.randint(1,min(20,len(available_solar_capacities)))]
		else: # Handle the case when available_solar_capacities has 1 item    	
			selectedCols = colChoice[0:1] 

		total_solar_power = pd.to_numeric(df_timeseries.loc[:, selectedCols].sum(axis=1),downcast='float') #float32, 4-bytes
		total_gross_load = pd.to_numeric(df_timeseries.loc[:,[entry.replace('solar_power','gross_load') for entry in selectedCols]].sum(axis=1),downcast='float') #float32, 4-bytes
		solar_scaling_factor = (P[node]*solar_node_dict[node]["solar_penetration"])/total_solar_power.values.max() #Calculate scaling factor for solar power
		gross_load_scaling_factor = P[node]/total_gross_load.values.max()
		
		solar_node_dict[node].update({"P":P[node]})
		solar_node_dict[node].update({"solar_scaling":solar_scaling_factor})
		solar_node_dict[node].update({"gross_load_scaling":gross_load_scaling_factor})
		solar_node_dict[node].update({"solar_power":total_solar_power.values})
		solar_node_dict[node].update({"gross_load":total_gross_load.values})

		res[f"{node}_solar"] = total_solar_power.values*solar_scaling_factor
		res[f"{node}_gross_load"] = total_gross_load.values*gross_load_scaling_factor
		res[f"{node}_net_load"] = res[f"{node}_gross_load"] - res[f"{node}_solar"]

	df_node = pd.DataFrame(res)
	print(f"Finished generating load profiles (each with {len(df_node)} time steps) for {len(solar_node_dict)} nodes...")	
	return df_node,solar_node_dict #### TODO replace solar_node_dict
