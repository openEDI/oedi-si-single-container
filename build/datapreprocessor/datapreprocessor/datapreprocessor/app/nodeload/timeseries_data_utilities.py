"""'
Created on Thursday Feb 16 15:00:00 2023
@author: Siby Plathottam
"""

import os
import json
import statistics

import pandas as pd

def get_time_series_dataframe(file_name):
	"""Define function to extract data from file, and return pandas dataframe."""
	
	with open(file_name) as f:
		first_line = f.readline()
	
	if 'block_index' in first_line: #Check if 'block_index' is in the list of column names
		df_timeseries = pd.read_csv(file_name,index_col='block_index')
	else:
		df_timeseries = pd.read_csv(file_name)
		print("Adding index name since it was not found!")
		df_timeseries.index.name = 'block_index'
	
	return df_timeseries

def add_datetime_and_save(time_series_files,index_label='block_index'):
    """Add datetime"""
    
    for file in time_series_files:
        df = pd.read_csv(file,index_col=0)
        try:
            df = add_datetime(df)
            print(f"Adding datetime to file:{file}")
            df.to_csv(path_or_buf =file,index_label=index_label)
        except AssertionError as error:
            print(f"Not adding datetime to file:{file} due to:{error}")

def find_unique_loads(column_names):
	
	unique_loads = set()

	pattern1 = r'customer_\d+_(\d+\.\d+)_' #Pattern for solar home data
	pattern2 = r'Load_\w+_'	  #Pattern for time series data

	for col in column_names:
		match1 = re.search(pattern1, col)
		match2 = re.search(pattern2, col)
		if match1:
			unique_loads.add(float(match1.group(1)))
		elif match2:
			unique_loads.add(match2.group(0)[:-1])

	print(f"Found following {len(unique_loads)} unique_loads:")
	
	return unique_loads

def add_datetime(df,show_details=False):
	"""Add datetime column"""
	
	assert 'datetime' not in df.columns, "datetime already in dataframe columns"
	assert 'date_block' in df.columns and 'time_block' in df.columns, "Check if date_block and time_block in dataframe columns"
	
	print(f"Adding datetime column...")
	if any('24:' in time_str for time_str in df['time_block'].to_list()): #check if time stamps contain 24 (hour must be in 0..23):
		print("Shift elements in time_block column by 1 to remove 24 from time_block strings")
		df['time_block'] = df['time_block'].shift(1)#Shift to fix timestamp hour format	
		df.loc[0,'time_block'] = '00:00:00' #Fill missing value due to shift
		df['time_block'] = df['time_block'].str.replace("24:","00:") #replace 24 with 00 at start of new day
	
	df.insert(0, 'datetime', df['date_block'] + "-" + df['time_block']) #Insert date time block
	df['datetime']	= pd.to_datetime(df['datetime'], format='%Y-%m-%d-%H:%M:%S')
	if show_details:
		print(df.head())
		print(df.tail())
	return df

def get_config_dict(config_file):
	"""Read JSON configuration file"""
	
	assert ".json" in config_file, f"{config_file} should be JSON file!"
	if not os.path.exists(config_file):
		raise ValueError(f"{config_file} is found!")
	else:
		print(f"Reading following config file:{config_file}")
	
	f = open(config_file)
		
	return json.load(f)

def combine_time_series_loadtype_months(load_type,year,months,zip_code,folder):
	"""Combine multiple time sereis files for a zipcode"""
	
	selected_time_series_files = []
	found_months = []
	for month in months:
		month = str(month).zfill(2)
		time_series_file = f"{year}_{month}_{zip_code}_time_series.csv"
		#print(f"Adding file {time_series_file}")
		if os.path.isfile(os.path.join(folder,time_series_file)):
			found_months.append(int(month))
			print(f"Adding file {time_series_file}")
			selected_time_series_files.append(time_series_file)
	#df = pd.concat((pd.read_csv(os.path.join(folder,time_series_file)) for time_series_file in selected_time_series_files))
	print(f"Found following {len(found_months)} months:{found_months} out of requested:{months} - missing:{set(months).difference(set(found_months))}")
	dfs = []
	dfs_date = []
	stats_dict = {}
	for time_series_file in selected_time_series_files:
		df = get_time_series_dataframe(os.path.join(folder,time_series_file))
		month_string = time_series_file.split("_")[1]
		print(f"Calcuating stats for {time_series_file}...")
		stats_dict.update({month_string:{"average":round(df.filter(regex = f"(?i){load_type}").mean().mean(),3)}})
		
		outage_dict = get_outages(df.filter(regex = f"(?i){load_type}"))
		stats_dict[month_string].update({"total_outages":outage_dict["total"],"outage_hours_per_customer":outage_dict["outage_hours_per_customer"]})
		
		#print(stats_dict)
		dfs_date.append(df.filter(items = ['block_index','date_block','time_block']))
		dfs.append(df.filter(regex = f"(?i){load_type}"))
		print(f"Customers:{len(dfs[-1].columns)},Time steps:{len(dfs[-1])}")
		#display(dfs[-1].head())
	
	stats_dict.update({f"average_of_{len(selected_time_series_files)}_months":statistics.mean([stats_dict[month]['average'] for month in stats_dict.keys()])})
	#stats_dict.update({f"total_outages_per_customer_in_{len(selected_time_series_files)}_months":sum([stats_dict[month]['total_outages'] for month in stats_dict.keys()])})
	#stats_dict.update({f"average_outages_per_customer_in_{len(selected_time_series_files)}_months":statistics.mean([stats_dict[month]['outage_hours_per_customer'] for month in stats_dict.keys()])})

	df = pd.concat(dfs,ignore_index =True)
	df = pd.concat([pd.concat(dfs_date,ignore_index =True),df],axis = 1)
	df['time_block'] = df['time_block'].shift(1) #Shift to fix timestamp hour format
	df.loc[0,'time_block'] = '00:00:00' #Fill missing value due to shift
	df.insert(0, "datetime", df['date_block'] + "-" + df['time_block']) #Insert date time block
	df['datetime'] = df['datetime'].str.replace("-24:","-00:")
	df['datetime']	= pd.to_datetime(df['datetime'], format='%Y-%m-%d-%H:%M:%S')
	
	return df,stats_dict

def get_statistics(df_dict,resample=False,resample_time = "60Min"):
	
	for zipcode in df_dict.keys():
		for load_type in df_dict[zipcode].keys():
			print(f"Calculating statistics for {zipcode}:{load_type}")
			df_timeseries = df_dict[zipcode][load_type]["df"]
			outage_dict = get_outages(df_timeseries) 
			df_dict[zipcode][load_type].update({"total_outages":outage_dict["total"]})
			df_dict[zipcode][load_type].update({"total_customers":len(df_timeseries.columns)-3})
			df_dict[zipcode][load_type].update({"outages_per_customer":df_dict[zipcode][load_type]["total_outages"]/df_dict[zipcode][load_type]["total_customers"]})
			
			if resample:
				print(f"Resampling to {resample_time} before calculating average...")
				df_timeseries = df_timeseries.set_index("datetime")
				average_energy_consumption = df_timeseries.resample(resample_time,label='left').sum().mean().mean()#convention='start')
				df_dict[zipcode][load_type].update({"average_energy_consumption":df_timeseries.resample(resample_time,label='left').sum().mean().mean()})
				df_dict[zipcode][load_type].update({"resample_time":resample_time})
			else:
				df_dict[zipcode][load_type].update({"average_energy_consumption":df_dict[zipcode][load_type]["df"].mean().mean()})
				df_dict[zipcode][load_type].update({"resample_time":"15Min"})
				df_dict[zipcode][load_type].update({"average_energy_consumption_corrected":df_dict[zipcode][load_type]['stats']['average_of_6_months']})
	
	for zipcode in df_dict.keys():
		for load_type in df_dict[zipcode].keys():
			print(f"Statistics for {zipcode} - {load_type}")
			print(f"Total consumers:{df_dict[zipcode][load_type]['total_customers']}")
			print(f"Average energy consumption:{df_dict[zipcode][load_type]['average_energy_consumption']:.3f}")
			print(f"Average energy consumption-corrected:{df_dict[zipcode][load_type]['average_energy_consumption_corrected']:.3f}")
			print(f"Outage hours per customer:{df_dict[zipcode][load_type]['outages_per_customer']:.0f}")
			
	return df_dict

def get_outages(df,time_interval_in_hour=0.5):
	"""Get outages by customer"""
	
	#df = df.drop(['date_block','time_block'],axis = 1)
	#display(df)
	outage_dict = {}
	print(f"Expected outage hours:{((df == 0.0).sum().sum())*time_interval_in_hour}")
	for column in list(df.columns):
		if column not in ['date_block','time_block']:
			outages = (df[column] == 0.0).sum()*time_interval_in_hour
			if outages > 0:	
				outage_dict.update({column:outages})
				#print(f'{outages} outages found in {column}')
				#display(df[['date_block','time_block']][df[column] == 0.0])
		
	total_outages = sum(list(outage_dict.values()))
	n_customers = len(df.columns)-2
	outage_hours_per_customer = round(total_outages/n_customers,2)
	print(f"Found outage hours:{total_outages}")
	outage_dict.update({'total':total_outages})
	outage_dict.update({'n_customers':n_customers})
	outage_dict.update({'outage_hours_per_customer':outage_hours_per_customer})
	
	return outage_dict

def get_n_days_in_df(df,datetime_column = "datetime"):
	# group the dataframe by month
	n_days = 0
	df_month = df.groupby(pd.Grouper(key=datetime_column, freq='M'))

	# calculate the number of days for each month
	for name, group in df_month:
		n_days = n_days + (group[datetime_column].max() - group[datetime_column].min()).days + 1
		print(f"Month: {name.month}, Number of days: {(group[datetime_column].max() - group[datetime_column].min()).days + 1}")
	
	return n_days

def add_milliseconds(df,datetime_column):
    df[datetime_column]= pd.to_datetime(df[datetime_column], format='%Y-%m-%d %H:%M:%S')
    df[datetime_column] = df[datetime_column] + pd.to_timedelta((df['fractional_seconds']-1)*100, unit='ms')
    
    return df
