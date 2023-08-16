"""'
Created on Wed June 01 00:10:00 2022
@author: splathottam
"""

import glob
import math
import calendar
import os
import pickle
import warnings
from typing import List, Set, Dict, Tuple, Optional, Union
from statistics import mean

import numpy as np
import pandas as pd
import tensorflow as tf
from numpy.random import default_rng
from pandas.tseries.offsets import DateOffset
from tqdm import tqdm

from datapreprocessor.app.dopf.model.distmodel import DistModel
from datapreprocessor.app.nodeload.timeseries_data_utilities import get_time_series_dataframe
from datapreprocessor.app.nodeload.nodeload_preprocessing import encode_cyclical_features
from datapreprocessor.app.nodeload.datapipeline_utilities import get_input_target_dataset

def get_corrupted_df(df,corrupt_value_replacement=-1.0,corrupted_fraction = 0.01,replacement_methods=[]):
	df["data_quality"] = "nominal"
	df["corruption_encoding"] = 0
	df["load_value_corrupted"] = df['load_value'].sample(frac=1-corrupted_fraction)	  #Corrupt a fraction of the dataframe rows using NAN
	
	total_corrupted_values = df['load_value_corrupted'].isnull().sum()
	print(f"Number of corrupted values time stamps:{total_corrupted_values}")#.any()}")
	corrupted_indexes = df[df['load_value_corrupted'].isnull()].index.tolist()
	
	df.loc[corrupted_indexes,"data_quality"] = "corrupted" #assign corrupted identifier to bad data
	df.loc[corrupted_indexes,"corruption_encoding"] = 1 #assign 1 as categorical encoding for corrupted data
	
	assert total_corrupted_values == len(corrupted_indexes), "Total corrupted indexes should be equal"
	
	
	df = get_replace_nans(df,replacement_methods=replacement_methods)
		
	df["load_value_corrupted"] = df["load_value_corrupted"].fillna(corrupt_value_replacement) #Fill NAN values with a scalar replacement
		
	#assert len(df[df['load_value_corrupted']==corrupt_value_replacement]) == total_corrupted_values, "Total replaced values should be equal to total corrupted values"
	assert len(df[df['data_quality']=="corrupted"]) == total_corrupted_values, "Total corrupt identifiers should be equal to total corrupted values"	
	
	return df,corrupted_indexes

def get_corrupted_df_multi(df,corrupt_value_replacement=-1.0,corrupted_fraction = 0.05,consequtive_event_probabilities={},replacement_methods=[]):
	"""Corrupt at either single or consequtive time stamps"""
	#print(consequtive_event_probabilities)
	rng = default_rng()
	probability_of_one_missing_value = corrupted_fraction #Probability of atleast one missing value
	
	if not consequtive_event_probabilities:
		consequtive_event_probabilities = {"two":{"conditional_probability":0.2},"three":{"conditional_probability":0.1}} #Here probability is conditional probabiliy of two consequtive missing value given one missing value has occured P(one:two)
	
	print(f"Probability of atleast one missing value:{probability_of_one_missing_value}")
	previous_event_probability = probability_of_one_missing_value
	for j,event in enumerate(consequtive_event_probabilities.keys()):
		consequtive_event_probabilities[event].update({"n_events":0})
		print(f"Probability of {event} consequtive missing values:{previous_event_probability*consequtive_event_probabilities[event]['conditional_probability']:.4f}")
		previous_event_probability = previous_event_probability*consequtive_event_probabilities[event]["conditional_probability"]
	
	one_missing_value_event = 0
	
	df["data_quality"] = "nominal" #assign corrupted identifier to bad data
	df["corruption_encoding"] = 0
	df["load_value_corrupted"] = df['load_value'] #initially no values are missing
	
	for i in tqdm(range(len(df))):
		if df.loc[i, "data_quality"] == "nominal":
			if rng.uniform(0,1.0) <= probability_of_one_missing_value:
				df.loc[i, "data_quality"] = "corrupted" #assign corrupted identifier to bad data
				one_missing_value_event = one_missing_value_event + 1
				previous_event_occured = True
				for j,event in enumerate(consequtive_event_probabilities.keys()):
					if rng.uniform(0,1.0) <= consequtive_event_probabilities[event]["conditional_probability"] and previous_event_occured:
						previous_event_occured = True
						df.loc[i+j+1, "data_quality"] = "corrupted" #assign corrupted identifier to bad data
						consequtive_event_probabilities[event]["n_events"] = consequtive_event_probabilities[event]["n_events"] +1
						#print(f"{event} consequtive missing value event at:{i,i+j+1}")
					else:
						previous_event_occured = False
	
	total_events = one_missing_value_event
	print(f"Event at least one count:{one_missing_value_event}")
	print(f"Event probability:one:{one_missing_value_event/len(df):.4f}")
	for j,event in enumerate(consequtive_event_probabilities.keys()):
		total_events = total_events + consequtive_event_probabilities[event]["n_events"]
		print(f"Event at least {event} count:{consequtive_event_probabilities[event]['n_events']}")
		print(f"Event probability:{event}:{consequtive_event_probabilities[event]['n_events']/len(df):.4f}")
	print(f"Total missing value events:{total_events}")
	corrupted_indexes = df[df['data_quality']=="corrupted"].index.tolist()
	
	df.loc[corrupted_indexes,"load_value_corrupted"] = np.nan #Corrupt a fraction of the dataframe rows using NAN
	df.loc[corrupted_indexes,"corruption_encoding"] = 1 #assign 1 as categorical encoding for corrupted data
	total_corrupted_values = df['load_value_corrupted'].isnull().sum()
	print(f"Number of corrupted values time stamps:{total_corrupted_values}")#.any()}")
	
	assert total_corrupted_values == len(corrupted_indexes), "Total corrupted indexes should be equal"
	
	df = get_replace_nans(df,replacement_methods=replacement_methods)		 
	df["load_value_corrupted"] = df["load_value_corrupted"].fillna(corrupt_value_replacement) #Fill NAN values with a scalar replacement
		
	assert len(df[df['data_quality']=="corrupted"]) == total_corrupted_values, "Total corrupt identifiers should be equal to total corrupted values"	
	
	return df,corrupted_indexes

def get_replace_nans(df,replacement_methods=["ffill","bfill","mean","median","LI"]):
	"""Corrupted data is replaced"""
	#replacement_methods=["ffill","bfill","mean","median","LI"]
	for replacement_method in replacement_methods:
		print(f"FIlling NAN values using {replacement_method}...")
		if replacement_method in ["ffill","bfill"]:
			df[f"load_value_corrupted_{replacement_method}"] = df["load_value_corrupted"].fillna(method = replacement_method) #Fill NAN values using forward fill
		elif replacement_method == "mean":
			df[f"load_value_corrupted_{replacement_method}"] = df["load_value_corrupted"].fillna(value=df["load_value_corrupted"].mean())
		elif replacement_method == "median":
			df[f"load_value_corrupted_{replacement_method}"] = df["load_value_corrupted"].fillna(value=df["load_value_corrupted"].median())
		elif replacement_method == "LI":			
			df[f"load_value_corrupted_{replacement_method}"] = df['load_value_corrupted'].interpolate(method='linear') #Fill nan with linear interpolation
		else:
			raise ValueError(f"Error in {replacement_method}")
			
		if df[f"load_value_corrupted_{replacement_method}"].isnull().values.any(): #df.loc[df[f"load_value_corrupted_{replacement_method}"].isnull().values,:].index.any():
			print(f"Found NAN when using {replacement_method}.... replacing with mean:{df['load_value_corrupted'].mean()}")
			print(df.loc[df[f"load_value_corrupted_{replacement_method}"].isnull().values,:])
			df[f"load_value_corrupted_{replacement_method}"] = df["load_value_corrupted"].fillna(value=df["load_value_corrupted"].mean())			 
			
		if df[f"load_value_corrupted_{replacement_method}"].isnull().values.any():
			raise ValueError(f"NAN values found in df after {replacement_method}!")
		
	return df

def add_anomaly_values(df,anomaly_value_types=[-1]):
	"""Add anomaly values to dataframe"""
	rng = default_rng()
	df["load_value_anomaly"] = df["load_value_corrupted"]
	
	for row in df.loc[df["corruption_encoding"]==1].iterrows():
		#print(row[0])
		df.loc[row[0],"load_value_anomaly"] = rng.choice(anomaly_value_types)
		
	return df

def get_comparison_df(df_source,predictions):
	df_comparison = pd.DataFrame()
	df_comparison["datetime"] = df_source["datetime"].values
	df_comparison["data_quality"] = df_source["data_quality"].values
	df_comparison["load_value_actual"] = df_source["load_value"].values
	
	df_comparison["load_value_bfill"] =df_source["load_value_corrupted_bfill"].values	 
	df_comparison["load_value_ffill"] =df_source["load_value_corrupted_ffill"].values
	df_comparison["load_value_mean"] =df_source["load_value_corrupted_mean"].values
	df_comparison["load_value_LI"] =df_source["load_value_corrupted_LI"].values
	
	df_comparison["load_value_DAE_prediction"] = predictions.flatten()
	
	df_comparison["prediction_bfill_AE"] = abs(df_comparison["load_value_actual"]-df_comparison["load_value_bfill"])
	df_comparison["prediction_ffill_AE"] = abs(df_comparison["load_value_actual"]-df_comparison["load_value_ffill"])
	df_comparison["prediction_DAE_AE"] = abs(df_comparison["load_value_actual"]-df_comparison["load_value_DAE_prediction"])
	
	df_comparison["prediction_bfill_SE"] = np.square(df_comparison["load_value_actual"].values-df_comparison["load_value_bfill"].values)	
	df_comparison["prediction_ffill_SE"] = np.square(df_comparison["load_value_actual"].values- df_comparison["load_value_ffill"].values)
	df_comparison["prediction_LI_SE"] = np.square(df_comparison["load_value_actual"].values- df_comparison["load_value_LI"].values)
	df_comparison["prediction_DAE_SE"] = np.square(df_comparison["load_value_actual"].values-df_comparison["load_value_DAE_prediction"].values)
	df_comparison["prediction_mean_SE"] = np.square(df_comparison["load_value_actual"].values-df_comparison["load_value_mean"].values)	  
	
	return df_comparison

def get_dataset_from_csv(time_series_files,load_type_selected,load_block_length,selected_month,input_features,target_feature,cyclical_features,
						 show_details=False,df_type="train",corrupted_fraction=0.01,use_moving_window=False,replacement_methods=[]):
	"""Helper method to create dataset"""
	
	df = get_df_from_timeseries_file(time_series_files=time_series_files,load_type = load_type_selected,load_block_length = load_block_length,selected_month=selected_month,show_details=show_details,cyclical_features=cyclical_features)
	df,corrupted_indexes = get_corrupted_df(df,corrupt_value_replacement=0.0,corrupted_fraction = corrupted_fraction,replacement_methods=replacement_methods)
	
	print(f"NAN in {df_type} df:{df.isnull().values.any()}")
	if df.isnull().values.any():
		df[df.isnull().values]#.values.any()
	#print(df[input_features].isnull().values.any())
	#df_train.loc[corrupted_indexes_train,:].head()
	
	input_dataset, target_dataset = get_input_target_dataset(df,load_block_length,input_features,target_feature,batch_size=None,use_moving_window=use_moving_window)
	
	print(f"Input dataset shape for {df_type}:{input_dataset.take(1).as_numpy_iterator().next().shape}")
	print(f"Target dataset shape for {df_type}:{target_dataset.take(1).as_numpy_iterator().next().shape}")
	input_target = tf.data.Dataset.zip((input_dataset, target_dataset))
	
	#train_input_target.take(1).as_numpy_iterator().next()[0].shape#[0:50].shape
	
	return df,input_target

def calc_imputation_performance(df_comparison):
	naive_mse = df_comparison[df_comparison["data_quality"]=="corrupted"]["prediction_mean_SE"].mean()
	bfill_mse = df_comparison[df_comparison["data_quality"]=="corrupted"]["prediction_bfill_SE"].mean()
	ffill_mse = df_comparison[df_comparison["data_quality"]=="corrupted"]["prediction_ffill_SE"].mean()
	LI_mse = df_comparison[df_comparison["data_quality"]=="corrupted"]["prediction_LI_SE"].mean()
	DAE_mse = df_comparison[df_comparison["data_quality"]=="corrupted"]["prediction_DAE_SE"].mean()
	
	bfill_mae = df_comparison[df_comparison["data_quality"]=="corrupted"]["prediction_bfill_AE"].mean()
	ffill_mae = df_comparison[df_comparison["data_quality"]=="corrupted"]["prediction_ffill_AE"].mean()
	DAE_mae = df_comparison[df_comparison["data_quality"]=="corrupted"]["prediction_DAE_AE"].mean()
	
	naive_rmse = naive_mse**0.5
	bfill_rmse = bfill_mse**0.5
	ffill_rmse = ffill_mse**0.5
	LI_rmse = LI_mse**0.5
	DAE_rmse = DAE_mse**0.5	   
	
	bfill_msse = bfill_mse/naive_mse
	ffill_msse = ffill_mse/naive_mse
	LI_msse = LI_mse/naive_mse
	DAE_msse = DAE_mse/naive_mse
	
	bfill_rmsse = bfill_rmse/naive_rmse
	ffill_rmsse = ffill_rmse/naive_rmse
	LI_rmsse = LI_rmse/naive_rmse
	DAE_rmsse = DAE_rmse/naive_rmse
	
	print(f"MAE - bfill:{bfill_mae:.3f},ffill:{ffill_mae:.3f},DAE:{DAE_mae:.3f}")
	print(f"MSE - bfill:{bfill_mse:.3f},ffill:{ffill_mse:.3f},DAE:{DAE_mse:.3f}")
	print(f"RMSE - bfill:{bfill_rmse:.3f},ffill:{ffill_rmse:.3f},DAE:{DAE_rmse:.3f}")
	print(f"MSSE - bfill:{bfill_msse:.3f},ffill:{ffill_msse:.3f},LI:{LI_msse:.3f},DAE:{DAE_msse:.3f}")
	print(f"RMSSE - bfill:{bfill_rmsse:.3f},ffill:{ffill_rmsse:.3f},LI:{LI_rmsse:.3f},DAE:{DAE_rmsse:.3f}")
	
	print(f"Relative improvement in MAE - bfill:{((bfill_mae-DAE_mae)/bfill_mae)*100:.2f},ffill:{((ffill_mae-DAE_mae)/ffill_mae)*100:.2f}")
	print(f"Relative improvement in RMSE - bfill:{((bfill_rmse-DAE_rmse)/bfill_rmse)*100:.2f},ffill:{((ffill_rmse-DAE_rmse)/ffill_rmse)*100:.2f},LI:{((LI_rmse-DAE_rmse)/LI_rmsse)*100:.2f}")
		
	return bfill_msse,ffill_msse,DAE_msse

def timeseries_to_pickle(time_series_files,load_type_selected,load_block_length,selected_month,input_features,target_feature,corrupted_fraction,pickle_file):
	"""Create pickle files with arrays that can be directly fed to data imputation model"""
	
	df,input_target = get_dataset_from_csv(time_series_files,load_type_selected,load_block_length,selected_month,input_features,target_feature,show_details=False,df_type="eval",corrupted_fraction=corrupted_fraction, use_moving_window= True)
	
	tfdataset_to_pickle(input_target,pickle_file)
	

def get_dataset_from_csv_v2(time_series_files,load_type_selected,load_block_length,selected_month,input_features,target_feature,cyclical_features,corrupted_fraction,
							show_details=False,df_type="train",use_moving_window= True,replacement_methods=[]):

	df = get_df_from_timeseries_file(time_series_files=time_series_files,load_type = load_type_selected,load_block_length = load_block_length,selected_month=selected_month,show_details=True,
				  cyclical_features=cyclical_features)
	df,corrupted_indexes = get_corrupted_df(df,corrupt_value_replacement=0.0,corrupted_fraction = corrupted_fraction,replacement_methods=replacement_methods)
	input_dataset = windowed_dataset(df[input_features], load_block_length)
	target_dataset = windowed_dataset(df[target_feature], load_block_length)
	input_target = tf.data.Dataset.zip((input_dataset, target_dataset))
	
	return input_target

def get_df_averaged_load_selected(df_averaged_load,load_type_selected,cyclical_features,selected_days,replacement_methods=[]):
	selected_days.sort()
	print(f"Selecting for day:{selected_days}")
	df_averaged_load_selected = df_averaged_load[["datetime","weekend",load_type_selected]]
	df_averaged_load_selected = df_averaged_load_selected.rename(columns={load_type_selected:target_feature})
	df_averaged_load = encode_cyclical_features(df_averaged_load_selected,cyclical_features,show_df=False,show_plot=False)
	df_averaged_load,corrupted_indexes = get_corrupted_df(df_averaged_load,corrupt_value_replacement=0.0,corrupted_fraction = corrupted_fraction,replacement_methods=replacement_methods)
	
	df_averaged_load = df_averaged_load[df_averaged_load["datetime"].dt.day.isin(selected_days)]
	
	return df_averaged_load

def get_df_node_load_selected_nodes(df_node_load,cyclical_features,selected_nodes,corrupted_fraction=0.01,multi_corruption=False,consequtive_corruption_probabilities={},replacement_methods=[]):
	selected_nodes.sort()
	n_timesteps = len(df_node_load)	
	print(f"Selected {len(selected_nodes)} load nodes containing {n_timesteps} time steps were selected:{selected_nodes[0:50]} (showing only first 50)")
	df_node_load_selected = df_node_load[["datetime"]+selected_nodes]
	df_train = pd.DataFrame()

	node_load_values = []
	node_load_ids = []
	node_load_time_stamps = []
	
	for node_id in selected_nodes:
		node_load_values.extend(df_node_load[node_id].values)
		node_load_time_stamps.extend(list(df_node_load["datetime"].values))
		node_load_ids.extend([node_id]*n_timesteps)
			
	df_train["datetime"] = node_load_time_stamps
	df_train["load_value"] = node_load_values
	df_train["node_id"] = node_load_ids
		
	df_train = encode_cyclical_features(df_train,cyclical_features,show_df=False,show_plot=False)	
	if corrupted_fraction > 0.0:
		if not multi_corruption:
			df_train,corrupted_indexes = get_corrupted_df(df_train,corrupt_value_replacement=0.0,corrupted_fraction = corrupted_fraction,replacement_methods=replacement_methods)
		else:
			df_train,corrupted_indexes = get_corrupted_df_multi(df_train,corrupt_value_replacement=0.0,corrupted_fraction = corrupted_fraction,consequtive_event_probabilities=consequtive_corruption_probabilities,replacement_methods=replacement_methods)
	else:
		df_train["data_quality"] = "nominal" #All data points are nominal
	
	return df_train

def get_knn_array(df,load_block_length,n_windows = 3000000):
	"""Get imputation from Scikit learn KNN imputation"""

	df["load_value_corrupted_nan"] = df["load_value_corrupted"]
	df.loc[df["corruption_encoding"]==1,"load_value_corrupted_nan"]=np.nan
	df[df["corruption_encoding"]==1].head()
	
	input_features_knn=	 ["load_value_corrupted_nan"] #KNN will only use one feature
	
	dataset_moving_window_knn,_ = get_input_target_dataset(df,load_block_length,input_features_knn,target_feature=None,batch_size=None,use_moving_window=True)

	array_moving_window_knn = np.array(list(dataset_moving_window_knn.take(n_windows).as_numpy_iterator()))
	print(f"Array shape from knn dataset:{array_moving_window_knn.shape}")
	array_moving_window_knn =	np.squeeze(array_moving_window_knn, axis=2)
	print(f"Array shape after reshape:{array_moving_window_knn.shape}")
	
	return array_moving_window_knn
