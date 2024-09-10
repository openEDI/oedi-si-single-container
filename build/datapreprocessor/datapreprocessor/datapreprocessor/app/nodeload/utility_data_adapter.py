"""
Orinally created on Wed May 01 00:10:00 2018
Restructured on May 14 2018
@author: splathottam
"""

import os
import glob
import zipfile
import random
import math
import warnings
from calendar import monthrange
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from tqdm import tqdm,trange

try:
	from bokeh.models import ColumnDataSource
	from bokeh.plotting import figure, show, output_file
	from bokeh.layouts import column
except ImportError:
	warnings.warn('bokeh failed to import', ImportWarning)

from datapreprocessor.datapreprocessor.app.nodeload.timeseries_data_utilities import get_time_series_dataframe

valid_load_types = ['Load_residential_single','Load_residential_multi','Load_residential_single_spaceheat','Load_residential_multi_spaceheat','Load_commercial','Load_0_100','Load_100_400','Load_400_1000','Load_misc']
valid_load_types_v2 = ['residential_single','residential_multi','residential_single_spaceheat','residential_multi_spaceheat','commercial','0_100','100_400','400_1000'] #'misc' #Removed Load suffix

class SmartMeterData():
	
		VERBOSE = False
		n_time_blocks = 48
		list_of_load_types = valid_load_types
		def __init__(self,folder_name_raw,folder_name_time_series,folder_name_zipped=None,plot_load_types=False,check_subfolders=False,show_details=True,unzip_zipped_files=True):
			
			self.master_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
			print('Current directory:',self.master_dir)
			self.folder_name_raw = folder_name_raw
			self.folder_name_time_series = folder_name_time_series
			self.folder_name_zipped = folder_name_zipped
			self.check_subfolders = check_subfolders
			self.update_folder_contents()
			if len(glob.glob(os.path.join(self.folder_name_zipped,'*.7z')))>0: #Check if there are any .7z zipped files and unzip them into the zipped folder
				extract_7z_files(self.folder_name_zipped,delete_after_extraction=True)
			if unzip_zipped_files:
				self.unzip_smartmeter_files(show_details)
			#if len(self.file_names_time_series)!=0:
				#self.count_load_types(plot_load_types)
			
		def update_folder_contents(self):
			"""Get contents inside folders."""
			
			if not self.check_subfolders:
				self.file_names_raw = glob.glob(os.path.join(self.folder_name_raw,'*.csv'))	 #List only Raw csv files
				self.file_names_time_series = glob.glob(os.path.join(self.folder_name_time_series,'*.csv'))	 #List only time series csv files					  
			else:
				self.file_names_raw = glob.glob(os.path.join(self.folder_name_raw,'*.csv'),recursive=True)	#List only Raw csv files
				self.file_names_time_series = glob.glob(os.path.join(self.folder_name_time_series,'*.csv'),recursive=True)	#List only time series csv files				
			
			if self.folder_name_zipped != None:
				if not self.check_subfolders:
					self.file_names_zipped = glob.glob(os.path.join(self.folder_name_zipped,'*.zip'))  #List only zip files
				else:					 
					self.file_names_zipped = glob.glob(os.path.join(self.folder_name_zipped,"**/*",'*.zip'),recursive=True)	 #List only zip files				 
			else:
				self.file_names_zipped = []
				
		def show_folder_contents(self,show_details=True):
			"""Show folders associated with object and its contents."""
			
			print('Number of raw smart meter files:{}'.format(len(self.file_names_raw)))
			print('Number of time series smart meter files:{}'.format(len(self.file_names_time_series)))
			print('Number of zipped smart meter files:{}'.format(len(self.file_names_zipped)))
			
			if show_details:				
				print('List of raw smart meter files...')
				for i,file_name in enumerate(self.file_names_raw,1):
					print('File {}:{}'.format(i,file_name))

				print('\nList of time series smart meter files...')
				for i,file_name in enumerate(self.file_names_time_series,1):
					print('File {}:{}'.format(i,file_name))

				print('\nList of zipped smart meter files...')
				for i,file_name in enumerate(self.file_names_zipped,1):
					print('File {}:{}'.format(i,file_name))
			
		def get_raw_dataframe(self,file_name):
			"""Define function to extract data from file, and return pandas dataframe."""
			
			df_input=pd.read_csv(file_name,parse_dates=['INTERVAL_READING_DATE'])

			df_input['weekday'] = df_input['INTERVAL_READING_DATE'].dt.dayofweek
			df_input['day'] =df_input['INTERVAL_READING_DATE'].dt.day
			df_input['month'] =df_input['INTERVAL_READING_DATE'].dt.month
			df_input['year'] =df_input['INTERVAL_READING_DATE'].dt.year
			df_input.drop(['TOTAL_REGISTERED_ENERGY','INTERVAL_LENGTH','ZIP_CODE','DELIVERY_SERVICE_CLASS',\
						   'ACCOUNT_IDENTIFIER','INTERVAL_HR2430_ENERGY_QTY','INTERVAL_HR2500_ENERGY_QTY'], axis=1,inplace=True)
			#df_input.drop('DELIVERY_SERVICE_NAME',axis=1,inplace=True),
			#print(df_input.info())
			#print(df_input.head())		  #Print first few columns of input data

			return df_input
		 
		def count_load_types(self,plot_pie=False,n_files=-1,timeseries_files=[],show_details=True):
						
			if len(timeseries_files) == 0:
				dataframe = get_time_series_dataframe(self.file_names_time_series[0])
				for x in trange(1, len(self.file_names_time_series[:n_files])):
					dataframe_x = get_time_series_dataframe(self.file_names_time_series[x])
					dataframe = pd.merge(dataframe,dataframe_x,on=['date_block', 'time_block'], how='left', suffixes=('','_{}'.format(x)))
			else:
				dataframe = get_time_series_dataframe(timeseries_files[0])
				x = 1
				for file_name in tqdm(timeseries_files[1:]):
					dataframe_x = get_time_series_dataframe(file_name)
					dataframe = pd.merge(dataframe,dataframe_x,on=['date_block', 'time_block'], how='left', suffixes=('','_{}'.format(x)))
					x = x+1
			
			self.residential_single_count = len(list(dataframe.filter(regex = self.list_of_load_types[0])))
			self.residential_multi_count = len(list(dataframe.filter(regex = self.list_of_load_types[1])))
			self.residential_single_spaceheat_count = len(list(dataframe.filter(regex = self.list_of_load_types[2])))
			self.residential_multi_spaceheat_count = len(list(dataframe.filter(regex = self.list_of_load_types[3])))
			self.commercial_count = len(list(dataframe.filter(regex = self.list_of_load_types[4])))
			self.load_0_100_count = len(list(dataframe.filter(regex = self.list_of_load_types[5])))
			self.load_100_400_count = len(list(dataframe.filter(regex = self.list_of_load_types[6])))			 
			self.misc_count = len(list(dataframe.filter(regex = self.list_of_load_types[7])))
			
			self.load_type_counts = [self.residential_single_count, self.residential_multi_count,
									 self.residential_single_spaceheat_count,self.residential_multi_spaceheat_count,self.commercial_count,self.load_0_100_count,self.load_100_400_count,self.misc_count]
			
			load_type_dict = {load_type:{} for load_type in self.list_of_load_types[0:8]}
			for load_type in self.list_of_load_types[0:8]:
				load_type_dict[load_type].update({"count":dataframe.filter(regex = load_type).count().count()})
				if load_type_dict[load_type]["count"]>0:
					load_type_dict[load_type].update({"mean":round(dataframe.filter(regex = load_type).mean().mean(),2)})
					load_type_dict[load_type].update({"max":round(dataframe.filter(regex = load_type).max().max(),2)})
					load_type_dict[load_type].update({"min":round(dataframe.filter(regex = load_type).min().min(),2)})
					load_type_dict[load_type].update({"mean_aggregate":round(dataframe.filter(regex = load_type).sum().mean(),1)})
			load_type_dict["Load_residential_single"]["count"] =  load_type_dict["Load_residential_single"]["count"]-load_type_dict["Load_residential_single_spaceheat"]["count"]
			load_type_dict["Load_residential_multi"]["count"] =	 load_type_dict["Load_residential_multi"]["count"]-load_type_dict["Load_residential_multi_spaceheat"]["count"]
			
			if show_details:
				print(f"Total expected customers columns:{len(dataframe.columns)-2}")
				for load_type in load_type_dict.keys():
					print(f"{load_type}-count:{load_type_dict[load_type]['count']}")
			total_loads_calculated = sum([load_type_dict[load_type]["count"] for load_type in load_type_dict.keys()])
			total_loads_expected = len(dataframe.columns)-3
			#assert total_loads_calculated==total_loads_expected,f"Calculated {total_loads_calculated} loads, expected:{total_loads_expected}"
			if not total_loads_calculated==total_loads_expected:
				print(f"Calculated {total_loads_calculated} loads, expected:{total_loads_expected}")
			
			#load_count_dict = {}
			if plot_pie==True:
				pie_chart_labels = []#['Load_residential_single', 'Load_residential_multi', 'Load_residential_single_spaceheat', 'Load_residential_multi_spaceheat','Load_commercial','Load_0_100','Load_100_400','Load_misc']
				pie_chart_counts = []
				pie_chart_means = []
				#for load_type,count in zip(self.list_of_load_types,self.load_type_counts): 
				#	 load_count_dict.update({load_type:count})
				for load_type in load_type_dict.keys():
					if load_type_dict[load_type]["count"] == 0:
						print(f"Removing {load_type}")						  
					else:
						pie_chart_counts.append(load_type_dict[load_type]["count"])
						pie_chart_means.append(load_type_dict[load_type]["mean"])
						pie_chart_labels.append(load_type)
				
				#fig = plt.figure(figsize =(10, 7))
				#plt.pie(pie_chart_counts, labels=pie_chart_labels, autopct='%1.1f%%', shadow=True)
				#plt.legend()
				#plt.show()				   
				
				x = np.char.array([load_type.replace("Load_","") for load_type in load_type_dict.keys()])
				y = np.array([load_type_dict[load_type]["count"] for load_type in load_type_dict.keys()])
				available_colors = ['brown','darkblue','gold','violet','green','orange','pink', 'darkgreen','yellow','grey','magenta','cyan']
				colors = [available_colors[i] for i in range(len(x)) ]
				porcent = 100.*y/y.sum()
				fig = plt.figure(figsize =(10, 7))				  
				patches, texts = plt.pie(y, colors=colors, startangle=90, radius=1.2,shadow=True)
				labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, porcent)]

				plt.legend(patches, labels, loc='center left', bbox_to_anchor=(-0.3, 1.),
						   fontsize=12)				   
				plt.savefig('piechart.png', bbox_inches='tight')
				plt.show()
				
				for load_type in load_type_dict.keys():
					if load_type_dict[load_type]["count"] > 0:						  
						for metric in ["mean","max"]:
							print(f"{load_type}:")
							print(f"Mean:{load_type_dict[load_type]['mean']}")
							print(f"Max:{load_type_dict[load_type]['max']}")
			
			return dataframe
		
		def get_date_zip_from_file_name(self,file_name):
			"""Function to extract year and month from smart metre data file."""
			
			temp_strings = file_name.replace(self.folder_name_raw,'').split('_')

			assert len(temp_strings)==4, 'wrong file name format!'
			date_string = temp_strings[2]
			zip_code = temp_strings[3].split('.')[0]
			year = date_string[0:4]
			month =date_string[4:6]

			return year,month,zip_code
		
		def unzip_smartmeter_files(self,show_details):
			"""Function to unzip smart meter file."""
			
			if len(self.file_names_zipped) >0:
				print('Number of zipped files:{}'.format(len(self.file_names_zipped)))
			else:
				print('No zipped files found!')
			
			print("Unzipping...")
			for file_name_zipped in tqdm(self.file_names_zipped):				
				if not os.path.isfile(os.path.join(self.folder_name_raw,file_name_zipped.split("/")[-1].replace(".zip",""))):  #Check if unzipped file exists
					if show_details:
						print(f'Unzipping {format(file_name_zipped)} to {self.folder_name_raw}')
					zip_ref = zipfile.ZipFile(file_name_zipped, 'r')
					zip_ref.extractall(self.folder_name_raw)
					zip_ref.close()
				else:
					if show_details:
						print(f'{format(file_name_zipped)} already unzipped to csv earlier!')
			
			self.update_folder_contents()

		def convert_to_timeseries(self,dataframe,days_in_month):
			"""Function to convert raw smart meter data to time series data."""
			
			k = 0
			i = 0
			
			dnew =pd.DataFrame()
			#Add time stamp
			date_time_index =0
			#Customer type count
			self.customer_residential_single = 0
			self.customer_residential_multi = 0
			self.customer_residential_single_spaceheat = 0
			self.customer_residential_multi_spaceheat = 0
			self.customer_commerical = 0
			self.customer_0_100 = 0
			self.customer_100_400 = 0
			self.customer_400_1000 = 0
			self.customer_misc = 0
			#print(days_in_month)
			for date in range(0,days_in_month):
				for time_block in range(0,self.n_time_blocks):
					dnew.loc[date_time_index,'date_block'] = pd.to_datetime(dataframe.loc[date,'INTERVAL_READING_DATE'])
					date_time_index = date_time_index+1	 
			time_index =0
			for date in range(0,days_in_month):
				for time_block in range(0,self.n_time_blocks):
					if (time_block%2) == 0:
						dnew.loc[time_index,'time_block'] = str(int((time_block+1)/2)).zfill(2)+':30:00'
					else:
						dnew.loc[time_index,'time_block'] = str(int((time_block+1)/2)).zfill(2)+':00:00'
					time_index = time_index+1  

			for customer_id in tqdm(range(0,int(len(dataframe)/days_in_month))):
				
				customer = self.get_customer_type(dataframe,customer_id*days_in_month)
				#print('Customer name:{}'.format(customer))
				for day in range(0,days_in_month):
					l = 0	
					for j in range(0,self.n_time_blocks):
						if (j%2) == 0:
							time_block = 'INTERVAL_HR'+str(l).zfill(2)	+'30'+'_ENERGY_QTY'
							
							l = l+1
						else:
							time_block = 'INTERVAL_HR'+str(l).zfill(2) +'00'+'_ENERGY_QTY'
							

						dnew.loc[k,customer] = dataframe.loc[i, time_block]
						k = k+1
						#print(k)

					i = i+1	 #Update index of old dataframe
				sequence_length = k				   
				k = 0	#Update index of time series dataframe
			#dnew.drop('DELIVERY_SERVICE_NAME',axis=1,inplace=True)
			print(f"Total loads:{customer_id+1},Total residential single loads:{self.customer_residential_single},Total residential multi loads:{self.customer_residential_multi},Total residential single space heat loads:{self.customer_residential_single_spaceheat},Total residential multi space heat loads:{self.customer_residential_multi_spaceheat},Total commercial:{self.customer_commerical},Total 0 - 100:{self.customer_0_100},Total 100 - 400:{self.customer_100_400},Total 400 - 1000:{self.customer_400_1000},Total misc loads:{self.customer_misc},Length of time series:{sequence_length}")
			if self.add_datetime:
				dnew = add_datetime(dnew,shift_time=True)
			if self.VERBOSE == True:
				print(dnew.head())
			return dnew, sequence_length

		def get_customer_type(self,dataframe,dataframe_index):
			"""Get customer type from raw file."""
			#print(dataframe.loc[dataframe_index,'DELIVERY_SERVICE_NAME'])
			#print(dataframe_index,dataframe.loc[dataframe_index,'INTERVAL_READING_DATE'],dataframe.loc[dataframe_index,'INTERVAL_HR0030_ENERGY_QTY'])
			if dataframe.loc[dataframe_index,'DELIVERY_SERVICE_NAME'] ==  'RESIDENTIAL SINGLE':				  
			   customer_type = 'load_residential_single_'+str(self.customer_residential_single)
			   self.customer_residential_single = self.customer_residential_single+1
			
			elif dataframe.loc[dataframe_index,'DELIVERY_SERVICE_NAME'] ==	'RESIDENTIAL MULTI':			   
			   customer_type = 'load_residential_multi_'+str(self.customer_residential_multi)
			   self.customer_residential_multi = self.customer_residential_multi+1
			
			elif dataframe.loc[dataframe_index,'DELIVERY_SERVICE_NAME'] ==	'RESIDENTIAL SINGLE (SPACE HEAT)':
			   customer_type = 'load_residential_single_spaceheat_'+str(self.customer_residential_single_spaceheat)
			   self.customer_residential_single_spaceheat = self.customer_residential_single_spaceheat+1
			
			elif dataframe.loc[dataframe_index,'DELIVERY_SERVICE_NAME'] ==	'RESIDENTIAL MULTI (SPACE HEAT)':
			   customer_type = 'load_residential_multi_spaceheat_'+str(self.customer_residential_multi_spaceheat)
			   self.customer_residential_multi_spaceheat = self.customer_residential_multi_spaceheat+1
			
			elif dataframe.loc[dataframe_index,'DELIVERY_SERVICE_NAME'] ==	'COM KWH ONLY':
			   customer_type = 'load_commercial_'+str(self.customer_commerical)
			   self.customer_commerical = self.customer_commerical+1
			
			elif dataframe.loc[dataframe_index,'DELIVERY_SERVICE_NAME'] ==	'SMALL (0 - 100)':#'DAILY (0-100) kWh':
			   customer_type = 'load_0_100_'+str(self.customer_0_100)
			   self.customer_0_100 = self.customer_0_100+1
			
			elif dataframe.loc[dataframe_index,'DELIVERY_SERVICE_NAME'] ==	'MED (100 - 400)': #'DAILY (100-400) kWh':
			   customer_type = 'load_100_400_'+str(self.customer_100_400)
			   self.customer_100_400 = self.customer_100_400+1
			elif dataframe.loc[dataframe_index,'DELIVERY_SERVICE_NAME'] ==	'LARGE (400 - 1000)': #'DAILY (400-1000) kWh':
			   customer_type = 'load_400_1000_'+str(self.customer_400_1000)
			   self.customer_400_1000 = self.customer_400_1000+1
			
			else:
			   customer_type = 'load_misc_'+str(self.customer_misc)
			   print(f"Unknown delivery service:{dataframe.loc[dataframe_index,'DELIVERY_SERVICE_NAME']}")
			   self.customer_misc = self.customer_misc+1
			
			return customer_type
	
		def convert_multiple_files(self,show_details=True,delete_raw_file=False):
			"""Function to convert multiple smart meter data files to time series."""
			
			if len(self.file_names_raw) > 0:
				print('Numer of CSV files found:{}'.format(len(self.file_names_raw)))

			else:
				print('No raw CSV files found!')
			
			for file_name in self.file_names_raw:  #Iterate over the list of file names
				
				year,month,zip_code =  self.get_date_zip_from_file_name(file_name) #Get days in month for current file
				days_in_month = monthrange(int(year), int(month))[1]
				
				save_path=os.path.join(self.folder_name_time_series,year+'_'+month+'_'+zip_code+'_time_series.csv')
				
				if not os.path.isfile(save_path):  #Check if time series file exists
					print(f'Converting raw smart meter data file - {file_name} - (Zip code: {zip_code}, Year: {year}, Month: {month}, Expected time series length:{days_in_month*48}) to time series...')
					df_raw =  self.get_raw_dataframe(file_name)
					df_timeseries,sequence_length = self.convert_to_timeseries(df_raw,days_in_month) #Convert to time series
					if sequence_length != days_in_month*self.n_time_blocks:
						print("Time series sequence length not matching with expected sequence length!")

					df_timeseries.to_csv(path_or_buf =save_path,index_label='block_index')
				else:
					if show_details:
						print(f"Converted file {save_path} already exists!")
				
				if delete_raw_file:
					if show_details:
						print(f"Removing raw file:{file_name}")
					os.remove(file_name)

			print('Number of converted files:{}'.format(len(self.file_names_time_series)))	   
			self.update_folder_contents()

		def validate_n_days(self,dataframe,file_name):
			"""Validate whether number of calculated days in dataframe is same as file name."""
			
			year,month,_ =	self.get_date_zip_from_file_name(file_name) #Get days in month for current file
			
			temp_strings = file_name.replace(self.folder_name_time_series,'').split('_')
			days_in_month = monthrange(int(year), int(month))[1]
			n_days_dataframe = self.get_n_days_dataframe(dataframe)
			
			if n_days_dataframe == days_in_month:
				print('Number of calculated days in dataframe is same as the month referenced by file name')
			else:
				print('Error: calculated days in dataframe is {} while days according to file name is {}'.format(n_days_dataframe,days_in_month))
		
		def compare_time_series_with_unzipped(self,year = 2017,month = 12,zip_code = '60654-5833'):
			"""Compare load types in time series files with those in corresponding raw files"""

			df_raw_test = pd.read_csv(os.path.join(self.folder_name_raw,f"ANONYMOUS_DATA_{year}{month}_{zip_code}.csv"))
			df_timeseries_test = pd.read_csv(os.path.join(self.folder_name_time_series,f"{year}_{month}_{zip_code}_time_series.csv"))
			
			unique_delivery_service_names = df_raw_test['DELIVERY_SERVICE_NAME'].unique()
			print(f"Found {len(unique_delivery_service_names)} Unique DELIVERY SERVICE NAMES:{unique_delivery_service_names}")
			load_names = [load_type for load_type in list(df_timeseries_test.columns) if load_type not in ['block_index','date_block','time_block']]
			#print(f"Actual load names:{load_names}")	

			unique_load_names = []
			for load_name in load_names:
				rc = ""
				for c in load_name.split("_")[0:-1]:
					rc = rc+"_"+c		 
				unique_load_names.append(rc[1:])
			print(f"Unique {len(unique_load_names)} load names:{list(set(unique_load_names))}")

		def get_n_days_dataframe(self,dataframe):
			"""Caclulate number of days in dataframe."""
			return int(len(dataframe)/self.n_time_blocks)
					  
		def get_random_dataframe(self):
			"""Get a random time series file."""
			_random_file = random.choice(self.file_names_time_series)
			
			df =  self.get_time_series_dataframe(_random_file)
			#self.validate_n_days(df,random_file)
			#file_name = _random_file.split('\\')[1]
			print('Random file name:{}'.format(_random_file.split('\\')[1]))
			return df,_random_file
		
		def get_random_day(self):
			"""Get random day corresponding to dataframe under test."""			  
			random_day = random.randint(1, self.n_days_test_dataframe)
			return random_day
		
		def get_day_starting_index(self,day):
			"""Get a starting index of specified day."""
			
			return (day-1)*self.n_time_blocks
		
		def get_day_ending_index(self,day,n_days):
			"""Get a ending index of specified day."""
			
			return (day+n_days-1)*self.n_time_blocks-1
		
		def check_start_day_validity(self,start_day,n_days):
			"""Check if given start day is valid for the dataframe under test."""
			
			if start_day == None:
				start_day = self.get_random_day()
				print("Starting day not given!- Using {}".format(start_day))
			if n_days == None:
				n_days = 1
				print("Number of days not given!- Using {}".format(n_days))
			
			assert start_day >0 and n_days <= self.n_days_test_dataframe, 'Start day must be greater than 0 and n_days must be lesser than number of days in dataframe!'
						
			if start_day + n_days-1 > self.n_days_test_dataframe:
					  start_day = self.n_days_test_dataframe-n_days+1
					  print("Starting day not valid!- Using {} as starting day".format(start_day))
			else:
				pass  #Do nothing if given start day is valid.
			
			return start_day
		
		def get_training_data(self,time_series_file_name=None,loads_to_include=['Load_residential_single'],load_list=None,
							  include_n_files=0,n_time_blocks=48):
			"""Return training data."""
			
			X_features = n_time_blocks
			print('Following loads will be included:',loads_to_include)
			loads_to_remove = list(set(self.list_of_load_types) - set(loads_to_include))  #Subtract loads to be included from list of load types
			if include_n_files>=len(self.file_names_time_series):
				include_n_files=len(self.file_names_time_series)
			if time_series_file_name==None:
				time_series_file_name = self.file_names_time_series[0]
			zip_info = []
			if load_list != None:
				dataframe = self.get_time_series_dataframe(time_series_file_name)
				for load in load_list:
					load_i = int(load.split('_')[-1])
					load_list[load_list.index(load)] = '{}_{}'.format(list(dataframe)[2].rsplit('_',1)[0],load_i)
				current_load_list = list(dataframe)
				loads_to_remove_x = list(set(current_load_list) - set(load_list) - set(['date_block', 'time_block']))
				dataframe.drop(columns=loads_to_remove_x,inplace=True)
				if list(dataframe)[-1].rsplit('_',1)[0] in set(loads_to_include):
					zip_info = self.file_names_time_series[0].split('_')[-3].split('-')[-1]
					zip_entries = (len(list(dataframe))-2)*(int(dataframe.count()[0]/48))
					zip_info = zip_entries*[zip_info]
			elif include_n_files<=0:
				dataframe = self.get_time_series_dataframe(time_series_file_name)
				if list(dataframe)[-1].rsplit('_',1)[0] in set(loads_to_include):
					zip_info = self.file_names_time_series[0].split('_')[-3].split('-')[-1]
					zip_entries = (len(list(dataframe))-2)*(int(dataframe.count()[0]/48))
					zip_info = zip_entries*[zip_info]
			else:
				dataframe = self.get_time_series_dataframe(self.file_names_time_series[0])
				if list(dataframe)[-1].rsplit('_',1)[0] in set(loads_to_include):
					zip_info = self.file_names_time_series[0].split('_')[-3].split('-')[-1]
					zip_entries = (len(list(dataframe))-2)*(int(dataframe.count()[0]/48))
					zip_info = zip_entries*[zip_info]
				for x in range(1, include_n_files):
					dataframe_x = self.get_time_series_dataframe(self.file_names_time_series[x])
					if list(dataframe_x)[-1].rsplit('_',1)[0] in set(loads_to_include):
						zip_info_x = self.file_names_time_series[x].split('_')[-3].split('-')[-1]
						zip_entries_x = (len(list(dataframe_x))-2)*(int(dataframe_x.count()[0]/48))
						zip_info_x = zip_entries_x*[zip_info_x]
						zip_info = zip_info + zip_info_x
					dataframe = pd.merge(dataframe,dataframe_x,on=['date_block', 'time_block'], how='left',suffixes=('','.{}'.format(x)))
				dataframe.drop_duplicates(subset=['date_block','time_block'],inplace=True)
			
			dataframe.drop(columns=['date_block', 'time_block'],inplace=True)
			customer_count = []
			for x in range(len(list(dataframe))):
				customer_count.append(int((dataframe.count()[x])/48))
			self.customer_count = customer_count
			#print(dataframe.head())
			for load_type in loads_to_remove:
				dataframe.drop(list(dataframe.filter(regex = load_type)), axis = 1, inplace = True)
			dataframe = dataframe.fillna(0)
			X_data = dataframe.values
			X_data.astype('float32', copy=False)
			
			X_data = X_data.flatten(order='F')
			
			X_data = X_data.reshape((int(len(X_data)/X_features),X_features))
			
			if X_data.size ==0:
				print('Empty array since dataset does not contain the specified loads to be included')
			for row_index in set(np.where(X_data==0)[0]):
				customer_count_index = math.floor(row_index/30)
				self.customer_count[customer_count_index] -= 1
			zip_info = np.asarray(zip_info, order='F')
			X_zip = np.concatenate((X_data,np.array([zip_info]).T), axis=1)
			X_zip = X_zip.astype('float')
			X_zip[X_zip == 0] = np.nan
			X_zip = X_zip[~np.isnan(X_zip).any(axis=1)]
			X_data = X_zip[:,:48]
			zip_info = X_zip[:,48:]
			zip_info = np.insert(zip_info, 0, int(self.file_names_time_series[0].rsplit('_')[-3].split('-')[0]), axis=1)
			
			return X_data,zip_info
		 
		def cluster(self, time_series_file_name=None, loads_to_include=['Load_residential_single'], load_list=None, include_n_files=0, 
					method='KMeans', plot_n_curves=0, n_clusters=4, n_time_blocks=48, visualization=False, scaled_clustering=False):
			if type(time_series_file_name) is int:
				time_series_file_name = [s for s in self.file_names_time_series if "{}".format(time_series_file_name) in s][0]
			#Get Data
			X_data,zip_info = self.get_training_data(time_series_file_name=time_series_file_name,
													 loads_to_include=loads_to_include,load_list=load_list,
													 include_n_files=include_n_files,n_time_blocks=n_time_blocks)
			
			#Scale Data
			if scaled_clustering==True:
				scaler = StandardScaler()
				scaler.fit(X_data)
				X_data_temp = scaler.transform(X_data)
			else:
				X_data_temp = X_data
			
			#Cluster Algorithm Selection and Get Centroids/Labels
			if method == 'KMeans':
				kmeans = KMeans(n_clusters=n_clusters).fit(X_data_temp)
				centroids = kmeans.cluster_centers_
				labels = kmeans.labels_
				if scaled_clustering==True:
					centroids = scaler.inverse_transform(centroids)
			elif method == 'Birch':
				brc = Birch(branching_factor=50, n_clusters=n_clusters, threshold=0.5,compute_labels=True)
				brc.fit(X_data_temp)
				labels = brc.labels_
				labeled_data = np.concatenate((X_data, np.array([labels]).T), axis=1)
				centroids = np.empty((0,49),float)
				for cluster in range(n_clusters):
					centroid_x = labeled_data[labeled_data[:,-1]==cluster]
					centroid_x = np.mean(centroid_x, axis=0, keepdims=True)
					centroids = np.append(centroids,centroid_x,0)
				centroids = np.delete(centroids,-1,1)
			else:
				print("Invalid clustering method. Valid options: 'KMeans','Birch'")
				return
			
			#Augment data
			labeled_data = np.concatenate((X_data, zip_info), axis=1)
			self.labeled_data = np.concatenate((labeled_data, np.array([labels]).T), axis=1)
			
			#Visualization
			if visualization==True:
				#If it is a local clustering, show df using customer count and labels
				if include_n_files==0:
					num_customers = self.customer_count
					dataframe = pd.DataFrame(data=np.zeros((max(labels)+1,len(num_customers))),
											 columns=['Customer {}'.format(i) for i in range(len(num_customers))], 
											 index=['Cluster {}'.format(j) for j in range(max(labels)+1)])
					labels_index = 0
					for i in range(len(num_customers)):
						for j in range(num_customers[i]):
							dataframe.loc['Cluster {}'.format(labels[labels_index]),'Customer {}'.format(i)] += 1
							labels_index += 1
					print(dataframe)
				self.plot_centroids(centroids)
				self.cluster_membership()
				if plot_n_curves>0:
					load_curves_to_plot = self.labeled_data[:plot_n_curves]
					self.plot_load_curves(load_curves_to_plot)
			
			return self.labeled_data,centroids
		
		def plot_centroids(self,centroids):
			"Plot the load curves that define each cluster centroid"
			plt.figure(figsize = (12,8))
			plt.xticks(list(range(0,48,4)), ('00:00', '02:00', '04:00', '06:00', '08:00', '10:00', '12:00', '14:00', '16:00',
											 '18:00', '20:00', '22:00'))
			plt.xlabel('Time of Day')
			#plt.yticks(list(np.arange(0,(np.amax(centers)+.05),.05)))
			plt.ylabel('Energy consumption (kWh)')
			plt.title('Characteristic Load Profiles of the {} clusters'.format(len(centroids)))
			colors = 10*(list('bgrcmk')+['b--', 'g--', 'r--', 'c--', 'm--', 'k--'])
			hh = list(range(48))
			for i in range(len(centroids)):
				plt.plot(hh, centroids[i], '{}'.format(colors[i]), label='Cluster {}'.format(i))
			plt.legend(loc=0)
			plt.show()
			return
		
		def plot_load_curves(self,labeled_data,load_list=None):
			"Plot several loads and color the curve based on cluster membership"
			n = len(labeled_data)
			hh = list(range(48))
			colors = 20*list('bgrcmk')
			plt.figure(figsize = (15,10))
			plt.xticks(list(range(0,48,4)), ('00:00', '02:00', '04:00', '06:00', '08:00', '10:00', '12:00', '14:00', '16:00',
											 '18:00', '20:00', '22:00'))
			plt.xlabel('Time of Day')
			#plt.yticks(list(np.arange(0,(np.amax(centers)+.05),.05)))
			plt.ylabel('Energy consumption (kWh)')
			if load_list == None:
				plt.title('Labeled Load Profiles for the first {} curves'.format(n))
			else:
				plt.title('Labeled Load Profiles for {}'.format(load_list))
			for i in range(len(labeled_data)):
				plt.plot(hh, labeled_data[i,:48], '{}'.format(colors[int(labeled_data[i,-1])]))
			plt.show()
			return
					
		def cluster_membership(self,labeled_data=np.full([1, 1], np.nan)):
			"Makes a pie chart of the sample size of each cluster"
			if np.isnan(labeled_data).any():
				labeled_data = self.labeled_data
			cluster_num = np.unique(self.labeled_data[:,-1], return_counts=True, axis=0)[0]
			cluster_counts = np.unique(self.labeled_data[:,-1], return_counts=True, axis=0)[1]
			plt.figure(figsize = (8,8))
			patches, texts, autotexts = plt.pie(cluster_counts, 
												labels=['Cluster {}'.format(int(num)) for num in cluster_num],
												autopct='%1.1f%%', shadow=True, colors=list('bgrcmk'))
			[ _.set_fontsize(22) for _ in texts]
			[ _.set_fontsize(22) for _ in autotexts]
			autotexts[0].set_color('y')
			if len(autotexts)>5:
				autotexts[5].set_color('y')
			plt.show()
			return
		
		def zip_codes_in_cluster(self,cluster_label,labeled_data=np.full([1, 1], np.nan),min_membership=2):
			"Lists out how much each zip code contributes to a certain cluster"
			if np.isnan(labeled_data).any():
				labeled_data = self.labeled_data
			cluster_x = labeled_data[labeled_data[:,-1]==cluster_label]
			cluster_num = np.unique(cluster_x[:,-3:-1], return_counts=True, axis=0)[0][:,1]
			cluster_counts = np.unique(cluster_x[:,-3:-1], return_counts=True, axis=0)[1]
			patches, texts, autotexts = plt.pie(cluster_counts,
												labels=['{}'.format(int(num)) for num in cluster_num],
												autopct='%1.1f%%', shadow=True, colors=list('bgrcmk'))
			plt.close()
			autotexts[0].set_color('y')
			if len(autotexts)>5:
				autotexts[5].set_color('y')
			for j in range(len(autotexts)):
				if float(autotexts[j].get_text()[:-1])>min_membership:
					print('{}-{}: {}'.format(int(self.file_names_time_series[0].rsplit('_')[-3].split('-')[0]),
												int(cluster_num[j]),autotexts[j].get_text()))
			return
		
		def clusters_in_zip_code(self,zip_code,labeled_data=np.full([1, 1], np.nan)):
			"Lists out how much each cluster contributes to a certain zip code"
			if np.isnan(labeled_data).any():
				labeled_data = self.labeled_data
			zip_x = labeled_data[labeled_data[:,-2]==zip_code]
			cluster_label = np.unique(zip_x[:,-1], return_counts=True, axis=0)[0]
			cluster_count = np.unique(zip_x[:,-1], return_counts=True, axis=0)[1]
			patches, texts, autotexts = plt.pie(cluster_count,
												labels=['{}'.format(int(num)) for num in cluster_label],
												autopct='%1.1f%%', shadow=True, colors=list('bgrcmk'))
			plt.close()
			autotexts[0].set_color('y')
			for j in range(len(autotexts)):
				if float(autotexts[j].get_text()[:-1])>0:
					print('Cluster {}: {}'.format(int(cluster_label[j]),autotexts[j].get_text()))
			return
		
		def pie_chart_of_cluster(self,cluster_num):
			"Makes a pie chart of each customer in a zip code's membership in a certain cluster"
			labels = self.labeled_data[:,-1].astype('int')
			num_customers = self.customer_count
			dataframe = pd.DataFrame(data=np.zeros((max(labels)+1,len(num_customers))),
									 columns=['Customer {}'.format(i) for i in range(len(num_customers))],
									 index=['Cluster {}'.format(j) for j in range(max(labels)+1)])
			labels_index = 0
			for i in range(len(num_customers)):
				for j in range(num_customers[i]):
					dataframe.loc['Cluster {}'.format(labels[labels_index]),'Customer {}'.format(i)] += 1
					labels_index += 1
			customers = list(dataframe.iloc[cluster_num,:].index.values)
			count = list(dataframe.iloc[cluster_num,:])
			i = 0
			for x in range(len(count)):
				if count[i]==0:
					count.remove(0)
					customers = np.delete(customers, i)
					i -= 1
				i += 1
			plt.figure(figsize = (8,8))
			patches, texts, autotexts = plt.pie(count,labels=customers,shadow=True)
			[ _.set_fontsize(22) for _ in texts]
			[ _.set_fontsize(22) for _ in autotexts]
			plt.show()
			return dataframe.iloc[cluster_num,:]
			
		def set_test_dataframe(self,dataframe,file_name):
			"""Define a data frame for testing."""
			self.test_file_name = file_name
			self.test_dataframe = dataframe
			self.n_days_test_dataframe = self.get_n_days_dataframe(dataframe)
		
		def info_test_dataframe(self):
			"""Define a data frame for testing."""
			print('File name corresponding to dataframe:{}'.format(self.test_file_name.split('\\')[1]))
			print('Number of loads:{}'.format(self.test_dataframe.shape[1]-2))
			print('Number of days:{}'.format(int(self.test_dataframe.shape[0]/self.n_time_blocks)))
		
		def plot_curves_dataframe(self,start_day=None,load_list=['Load_0','Load_1','Load_2'],return_dataframe=True):
			"""Plot load curve for specified day."""
			
			start_day = self.check_start_day_validity(start_day=start_day,n_days = 1)  #Curves for only one day are used
			start_index = self.get_day_starting_index(day=start_day)
			end_index = self.get_day_ending_index(day=start_day,n_days = 1)
			
			df = self.test_dataframe.loc[start_index:end_index][['time_block']+load_list]

			df.plot(x=['time_block'],y=load_list,title='Load curve for day {} starting at index {}'.format(start_day,start_index))
			
			if return_dataframe == True:
				return df
		
		def plot_histogram_dataframe(self,start_day=None,n_days=1,load_list=['Load_0','Load_1','Load_2'],return_dataframe = False):
			"""Plot stacked histogram for specified loads on specific days in the dataframe."""
			
			start_day = self.check_start_day_validity(start_day,n_days)
			start_index = self.get_day_starting_index(day=start_day)  #Computing starting and ending index of dataframe
			end_index = self.get_day_ending_index(day=start_day,n_days = n_days)
			
			df = self.test_dataframe.loc[start_index:end_index][load_list]
			_,load_mean,load_std_dev = self.calc_statistics_dataframe(df)
					  
			df.plot.hist(y=load_list,title='Histogram of loads for day {} to day {}\n mean:{}, standard deviation:{}'.format(start_day,start_day+n_days-1,load_mean,load_std_dev),stacked=True,alpha=0.75,figsize= (8,8))
			print(load_mean)
			for mean in load_mean:
				plt.axvline(mean, color = 'r', linestyle = 'dashed', linewidth = 2)
			
			if return_dataframe == True:
				return df
		
		def calc_statistics_dataframe(self,dataframe):
			"""Plot statistics for specified loads."""

			load_sum = dataframe.sum().as_matrix().astype('float32', copy=False)
			load_mean = dataframe.mean().as_matrix().astype('float32', copy=False)
			load_standard_deviation = dataframe.std().as_matrix().astype('float32', copy=False)
			
			return load_sum,load_mean,load_standard_deviation
		
		def plot_piechart_dataframe(self,start_day=None,n_days=1,load_list=['Load_0','Load_1','Load_2'],return_dataframe = False):
			"""Plot pie chart for specified loads."""
			
			start_day = self.check_start_day_validity(start_day,n_days)
			start_index = self.get_day_starting_index(day=start_day)  #Computing starting and ending index of dataframe
			end_index = self.get_day_ending_index(day=start_day,n_days = n_days)
			
			df = self.test_dataframe.loc[start_index:end_index][load_list]
						
			load_sum,_,_ = self.calc_statistics_dataframe(df)
			
			plt.pie(load_sum,labels =load_list)
			
			if return_dataframe == True:
				return df
			
		def explore_data_by_load_type_month(self,load_types,months,plot_folder):
			for selected_load_type in load_types:  #=['Load_residential_single', 'Load_residential_multi',]
				for selected_month in months:#[1,2,3,4,5,6,7,8,12]:
					timeseries_files_for_selected_month = [file for file in self.file_names_time_series if f"_{str(selected_month).zfill(2)}_" in file]
					print(f"Found {len(timeseries_files_for_selected_month)} files for {selected_load_type} in month:{selected_month}")
					df_explore = get_df_from_timeseries_file(timeseries_files_for_selected_month,load_type = selected_load_type,selected_month=selected_month,n_timesteps_per_day=48,show_details=False)
					if len(df_explore)>1:
						plot_distribution(df_explore,column_names=["load_value"],plot_type='box',fig_size=(10, 10),plot_folder=plot_folder,plot_title=f'{selected_load_type}_month-{selected_month}',show_plot=True,show_means=True,y_label = "kWh")
                        
class SmartMeterDataAnalytics(SmartMeterData):	
	"""Class to perform basic data analytics."""
	
	list_of_load_types = ['Load_residential_single','Load_residential_multi','Load_residential_spaceheat','Load_misc']
	def __init__(self, zip_code, start_date, end_date, plot_load_types=False):
		self.master_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir) 
		SmartMeterData.__init__(self,
							f'{self.master_dir}\\{zip_code}\\smartmeter_unzipped',
							f'{self.master_dir}\\{zip_code}\\smartmeter_timeseries',
							f'{self.master_dir}\\{zip_code}\\smartmeter_zipped',
							plot_load_types=plot_load_types)
		self.zip_code = zip_code
		self.start_date = start_date
		self.end_date = end_date
		self.macrostats_folder = os.path.join(f'{self.master_dir}\\{self.zip_code}','macrostats')
		if not os.path.exists(self.macrostats_folder):
			os.makedirs(self.macrostats_folder)
		return
		
	def calc_member_contribution(self,loads_to_include=['Load_residential_single','Load_residential_multi','Load_residential_single_spaceheat','Load_residential_multi_spaceheat'], 
								 plot_results=True):
		
		"""Calculate how many customers are recorded in a month range and the total energy contribution of each customer type"""
		
		macrostats_folder = self.macrostats_folder
		
		df_customer = pd.read_csv(os.path.join(macrostats_folder,'customers.csv')).set_index('date_time')
		df_customer.index = pd.to_datetime(df_customer.index)
		df_customer = df_customer.loc[self.start_date:self.end_date]
		index = []
		for date in df_customer.index:
			index.append(f'{date.year}-{date.month}')
		df_customer.index = pd.Series(index)
		df_customer = df_customer[['Residential Single', 'Residential Multi', 'Residential Single Spaceheat', 'Residential Multi Spaceheat']]
		
		
		df_single = pd.read_csv(os.path.join(macrostats_folder,'single.csv')).set_index('date_time')
		df_single.index = pd.to_datetime(df_single.index)
		df_multi = pd.read_csv(os.path.join(macrostats_folder,'multi.csv')).set_index('date_time')
		df_multi.index = pd.to_datetime(df_multi.index)
		df_single_spaceheat = pd.read_csv(os.path.join(macrostats_folder,'single_spaceheat.csv')).set_index('date_time')
		df_single_spaceheat.index = pd.to_datetime(df_single_spaceheat.index)
		df_single_spaceheat = pd.read_csv(os.path.join(macrostats_folder,'multi_spaceheat.csv')).set_index('date_time')
		df_multi_spaceheat.index = pd.to_datetime(df_single_spaceheat.index)
		total_consumption = {'Residential Single': df_single.loc[self.start_date:self.end_date]['Sum'].sum(),
							 'Residential Multi': df_multi.loc[self.start_date:self.end_date]['Sum'].sum(),
							 'Residential Single Spaceheat': df_single_spaceheat.loc[self.start_date:self.end_date]['Sum'].sum(),
							 'Residential Multi Spaceheat': df_single_spaceheat.loc[self.start_date:self.end_date]['Sum'].sum()}
		
		if plot_results==True:
			plt.figure(figsize = (20,12))
			df_customer.plot.bar(stacked=True, figsize = (20,12), title = "Number of Each Customer Type for Selected Month Range")
			plt.ylabel('Number of Customers')
			plt.xlabel('Month')
			plt.show()
			labels = []
			fracs = []
			for load_type in total_consumption:
				labels.append(load_type.split('_',1)[-1])
				fracs.append(total_consumption[load_type])
			plt.figure(figsize = (10,10))
			plt.pie(fracs, labels=labels, autopct='%1.1f%%', shadow=True)
			plt.title('Contribution to Total Energy Consumption')
			plt.show()
			
		return df_customer, total_consumption
		
	def define_selected_months(self):		
		"""Create a list of the month range based on the start and end date"""
		
		y = int(self.start_date.split("-")[0])
		m = int(self.start_date.split("-")[1])
		selected_months = [f'{y}_{m:02}']
		while((y!=int(self.end_date.split("-")[0]))|(m!=int(self.end_date.split("-")[1]))):
			m += 1
			if m==13:
				m = 1
				y += 1
			selected_months.append(f'{y}_{m:02}')
		return selected_months
	
	def compute_macrostats(self, delete_intermediate_files=True):		
		"""Calculate and save the average and total energy loads for each customer type"""
		
		macrostats_folder = self.macrostats_folder

		monthly_customers = {'Residential Single': OrderedDict(), 'Residential Multi': OrderedDict(), 
							 'Residential Spaceheat': OrderedDict()}
		encountered_months = []

		df_single = pd.DataFrame()
		df_multi = pd.DataFrame()
		df_spaceheat = pd.DataFrame()
		for file in self.file_names_time_series:
			file_month = file.rsplit('_',3)[0].rsplit('\\')[-1]
			df_month = self.get_time_series_dataframe(file)
			df_month.loc[df_month['time_block'] == '24:00:00', 'time_block'] = '23:59:59'
			df_month["date_time"] = pd.to_datetime(df_month.date_block+ " " + df_month.time_block, format = "%Y-%m-%d %H:%M:%S")
			df_month.drop(columns=['date_block', 'time_block'],inplace=True)
			df_month.drop_duplicates('date_time',inplace=True)
			df_month = df_month.set_index('date_time')
			df_single_x = df_month.filter(regex = 'Load_residential_single')
			df_single_x = df_single_x.replace(0,np.nan)
			df_single_result = df_single_x.mean(1).to_frame('Avg')
			df_single_result['Sum'] = df_single_x[df_single_x.columns].sum(axis=1)
			df_multi_x = df_month.filter(regex = 'Load_residential_multi')
			df_multi_x = df_multi_x.replace(0,np.nan)
			df_multi_result = df_multi_x.mean(1).to_frame('Avg')
			df_multi_result['Sum'] = df_multi_x[df_multi_x.columns].sum(axis=1)
			df_spaceheat_x = df_month.filter(regex = 'Load_residential_spaceheat')
			df_spaceheat_x = df_spaceheat_x.replace(0,np.nan)
			df_spaceheat_result = df_spaceheat_x.mean(1).to_frame('Avg')
			df_spaceheat_result['Sum'] = df_spaceheat_x[df_spaceheat_x.columns].sum(axis=1)

			single_customers = len(list(df_single_x))
			multi_customers = len(list(df_multi_x))
			spaceheat_customers = len(list(df_spaceheat_x))
			if file_month not in encountered_months:	
				monthly_customers['Residential Single'][file_month] = 0
				monthly_customers['Residential Multi'][file_month] = 0
				monthly_customers['Residential Spaceheat'][file_month] = 0
				encountered_months.append(file_month)
			monthly_customers['Residential Single'][file_month] += single_customers
			monthly_customers['Residential Multi'][file_month] += multi_customers
			monthly_customers['Residential Spaceheat'][file_month] += spaceheat_customers

			print(file)
			try:
				df_single = pd.concat([df_single,df_single_result],axis=1)
				df_multi = pd.concat([df_multi,df_multi_result],axis=1)
				df_spaceheat = pd.concat([df_spaceheat,df_spaceheat_result],axis=1)#,ignore_index=True
			except ValueError:
				continue
		df_single_avg = df_single.filter(regex = 'Avg').mean(1).to_frame('Avg')
		df_multi_avg = df_multi.filter(regex = 'Avg').mean(1).to_frame('Avg')
		df_spaceheat_avg = df_spaceheat.filter(regex = 'Avg').mean(1).to_frame('Avg')
		df_single_avg['Sum'] = df_single.filter(regex = 'Sum').sum(axis=1).divide(1000)
		df_multi_avg['Sum'] = df_multi.filter(regex = 'Sum').sum(axis=1).divide(1000)
		df_spaceheat_avg['Sum'] = df_spaceheat.filter(regex = 'Sum').sum(axis=1).divide(1000)

		df_monthly_customers = pd.DataFrame(monthly_customers)
		df_monthly_customers.index = pd.to_datetime(df_monthly_customers.index, format='%Y_%m')
		df_monthly_customers.index.name = 'date_time'

		df_monthly_customers.to_csv(os.path.join(macrostats_folder,'customers.csv'))
		df_single_avg.to_csv(os.path.join(macrostats_folder,'single.csv'))
		df_multi_avg.to_csv(os.path.join(macrostats_folder,'multi.csv'))
		df_spaceheat_avg.to_csv(os.path.join(macrostats_folder,'spaceheat.csv'))
		
		return
	
	def bokeh_plot_macrostats(self):
		
		"""Displays the average and total energy loads for each customer type"""
		
		macrostats_folder = self.macrostats_folder
		df_source_single = pd.read_csv(os.path.join(macrostats_folder,'single.csv')).set_index('date_time')
		df_source_single.index = pd.to_datetime(df_source_single.index)
		df_source_multi = pd.read_csv(os.path.join(macrostats_folder,'multi.csv')).set_index('date_time')
		df_source_multi.index = pd.to_datetime(df_source_multi.index)
		df_source_spaceheat = pd.read_csv(os.path.join(macrostats_folder,'spaceheat.csv')).set_index('date_time')
		df_source_spaceheat.index = pd.to_datetime(df_source_spaceheat.index)
		#Use bokeh library to plot
		source_single = ColumnDataSource(df_source_single)
		source_multi = ColumnDataSource(df_source_multi)
		source_spaceheat = ColumnDataSource(df_source_spaceheat)
		p = figure(x_axis_type="datetime", plot_width=900, plot_height=900, title="Average Energy Consumption for Zip Code: {}"
				  .format(self.zip_code))
		p.yaxis.axis_label = "Energy Consumption (kWh)"
		p.xaxis.axis_label = "Time of the Year"
		p.step('date_time', 'Avg', source=source_single, color='blue', legend='Residential Single')
		p.step('date_time', 'Avg', source=source_multi, color='green', legend='Residential Multi')
		p.step('date_time', 'Avg', source=source_spaceheat, color='orange', legend='Residential Spaceheat')
		p.legend.click_policy="hide"
		pt = figure(x_axis_type="datetime", plot_width=900, plot_height=900, title="Total Energy Consumption for Zip Code: {}"
				  .format(self.zip_code))
		pt.yaxis.axis_label = "Energy Consumption (MWh)"
		pt.xaxis.axis_label = "Time of the Year"
		pt.step('date_time', 'Sum', source=source_single, color='blue', legend='Residential Single')
		pt.step('date_time', 'Sum', source=source_multi, color='green', legend='Residential Multi')
		pt.step('date_time', 'Sum', source=source_spaceheat, color='orange', legend='Residential Spaceheat')
		pt.legend.click_policy="hide"
		show(column(p,pt))
		return

def get_load_code(load_type):
	assert load_type in valid_load_types, f"{load_type} is not a valid load type:{valid_load_types}"
	
	if load_type == "Load_residential_single":
		load_code = "rs"
	elif load_type == "Load_residential_multi":
		load_code = "rm"
	elif load_type == "Load_residential_single_spaceheat":
		load_code = "rssh"
	elif load_type == "Load_residential_multi_spaceheat":
		load_code = "rmsh"
	else:
		load_code = "misc"
	
	return load_code

def get_load_dict_legacy(df_timeseries,timeseries_file):
	"""Get load dict"""	
	
	load_type_col_names =[col_name for col_name in list(df_timeseries.keys()) if "load_" in col_name.lower()]
	print(f"Found {len(load_type_col_names)} loads in {timeseries_file}")

	load_dict.update({"residential_single":[col_name for col_name in load_type_col_names if "residential_single_" in col_name and '_spaceheat_' not in col_name]})
	load_dict.update({"residential_multi":[col_name for col_name in load_type_col_names if "residential_multi_" in col_name and '_spaceheat_' not in col_name]})
	load_dict.update({"residential_single_spaceheat":[col_name for col_name in load_type_col_names if "residential_single_spaceheat_" in col_name]})
	load_dict.update({"residential_multi_spaceheat":[col_name for col_name in load_type_col_names if "residential_multi_spaceheat_" in col_name]})
	load_dict.update({"commercial":[col_name for col_name in load_type_col_names if "commercial_" in col_name]})
	load_dict.update({"small_0_100":[col_name for col_name in load_type_col_names if "0_100_" in col_name]})
	load_dict.update({"medium_100_400":[col_name for col_name in load_type_col_names if "100_400_" in col_name]})
	load_dict.update({"large_400_1000":[col_name for col_name in load_type_col_names if "400_1000" in col_name]}) 
	print(f"Total identified loads:{sum(len(values) for values in load_dict.values())} in {timeseries_file}")
	
	return load_dict

def get_df_node_load_selected_samples(df_node_load,df_load_fraction,cyclical_features,selected_samples,corrupted_fraction=0.01,multi_corruption=False,replacement_methods=[]):
	selected_samples.sort()
	print(f"Selected {len(selected_samples)} samples:{selected_samples}")
	
	loads_selected = [f"node_load_{sample_id}" for sample_id in selected_samples]
	print(f"Following loads were selected:{loads_selected}")
	df_node_load_selected = df_node_load[["datetime"]+loads_selected]
	df_train = pd.DataFrame()

	node_load_values = []
	node_load_ids_time_series = []
	
	n_timesteps_per_day = len(df_node_load)
	node_load_time_stamps = list(df_node_load["datetime"].values)*len(loads_selected)
	node_load_fractions_time_series = {'residential_single':[], 'residential_multi':[], 'residential_single_spaceheat':[], 'residential_multi_spaceheat':[], 'commercial':[], 'small_0_100':[], 'medium_100_400':[]}
	
	for sample_id in selected_samples:
		node_load_values.extend(df_node_load_selected[f"node_load_{sample_id}"].values)
		node_load_ids_time_series.extend([sample_id]*n_timesteps_per_day)
		for load_type in ['residential_single', 'residential_multi', 'residential_single_spaceheat', 'residential_multi_spaceheat', 'commercial', 'small_0_100', 'medium_100_400']:
			node_load_fractions_time_series[load_type].extend([df_load_fraction.iloc[sample_id][load_type]]*n_timesteps_per_day)
	
	df_train["datetime"] = node_load_time_stamps
	df_train["load_value"] = node_load_values
	df_train["sample_id"] = node_load_ids_time_series
	for load_type in ['residential_single', 'residential_multi', 'residential_single_spaceheat', 'residential_multi_spaceheat', 'commercial', 'small_0_100', 'medium_100_400']:
		df_train[f"{load_type}_frac"] = node_load_fractions_time_series[load_type]
	
	df_train = encode_cyclical_features(df_train,cyclical_features,show_df=False,show_plot=False)
	if not multi_corruption:
		df_train,corrupted_indexes = get_corrupted_df(df_train,corrupt_value_replacement=0.0,corrupted_fraction = corrupted_fraction,replacement_methods=replacement_methods)
	else:
		df_train,corrupted_indexes = get_corrupted_df_multi(df_train,corrupt_value_replacement=0.0,corrupted_fraction = corrupted_fraction,replacement_methods=replacement_methods)
	
	return df_train
