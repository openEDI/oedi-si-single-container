"""'
Created on Wed September 16 9:00:00 2022
@author: Siby Plathottam
"""

import glob
import math
import calendar
import os
import pickle
import warnings
from typing import List, Set, Dict, Tuple, Optional, Union
from statistics import mean

import py7zr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from numpy.random import default_rng
from pandas.tseries.offsets import DateOffset
from tqdm import tqdm

from datapreprocessor.app.nodeload.datapipeline_utilities import get_input_target_dataset,check_moving_window

def evaluate_dataimputation_on_streaming_data_single_node(df_eval,selected_node,load_block_length):
	"""Evaluate on a datastream with missing values at time stamps"""
	
	#test_data_window = [df_node_load_eval[selected_node].mean()]*load_block_length
	test_data_window = [df_eval.loc[(df_eval["node_id"]==selected_node),"load_value"].mean()]*load_block_length
	test_data_ffill_window = [df_eval.loc[(df_eval["node_id"]==selected_node),"load_value"].mean()]*load_block_length
	actual_data_window = [df_eval.loc[(df_eval["node_id"]==selected_node),"load_value"].mean()]*load_block_length
	
	test_hour_window = [23,23,23,23]

	df_eval_selected_node = df_eval.loc[(df_eval["node_id"]==selected_node),["datetime","load_value","load_value_corrupted","load_value_corrupted_ffill"]]
	df_eval_selected_node = df_eval_selected_node.reset_index(drop=True)
	timestamp_window = [df_eval_selected_node.loc[0,"datetime"]]*load_block_length
	
	error_di_model = []
	error_ffill_model = []

	for row in range(len(df_eval_selected_node)):
		#df_row = df_node_load_eval.loc[row,["datetime",selected_node]]
		df_row = df_eval_selected_node.loc[row,:]
		#print(df_row)
		new_timestamp = df_row["datetime"]
		new_hour = df_row["datetime"].hour
		new_data = df_row.loc["load_value_corrupted"]
		new_data_ffill = df_row.loc["load_value_corrupted_ffill"]
		new_data_actual = df_row.loc["load_value"]

		window = create_streaming_data(new_data,new_data_ffill,new_hour,test_data_window,test_data_ffill_window,test_hour_window,load_block_length,input_features)

		actual_data_window.append(actual_data_window.pop(0)) #shifted backwards (to left)
		actual_data_window[-1] = new_data_actual #Replace with new data point
		timestamp_window.append(timestamp_window.pop(0)) #shifted backwards (to left)
		timestamp_window[-1] = new_timestamp #Replace with new data point

		if 0.0 in test_data_window:
			missing_index = test_data_window.index(0.0)
			prediction = autoencoder.predict(np.expand_dims(window, axis=0),verbose=0)
			#print(f"Missing data in moving window:{row} - Actual:{actual_data_window[missing_index]:.2f},AE prediction:{prediction.flatten()[missing_index]:.2f},ffill:{test_data_ffill_window[missing_index]:.2f}")
			print(f"Missing data at node:{selected_node},time stamp:{timestamp_window[missing_index]} in moving window:{row} - AE imputation:{prediction.flatten()[missing_index]:.2f},ffill:{test_data_ffill_window[missing_index]:.2f}")

			error_di_model.append((actual_data_window[missing_index] - prediction.flatten()[missing_index])**2)
			#error_ffill_model.append((actual_data_window[missing_index] - window[missing_index,1])**2)
			error_ffill_model.append((actual_data_window[missing_index] - test_data_ffill_window[missing_index])**2)

	print(f"DAE error:{sum(error_di_model)/len(error_di_model):.2f},ffill error:{sum(error_ffill_model)/len(error_ffill_model):.2f}")

def create_streaming_data_legacy(new_data,new_data_ffill,new_hour,data_window,data_ffill_window,hour_window,load_block_length,input_features):
	"""Returns a window that can be used as input to data imputation model that simulates a moving window"""
	
	#print(input_features)
	assert len(data_window)== len(hour_window) == load_block_length, f"Expected data window:{data_window} to have {load_block_length} elements"
	#data_window.append(data_window.pop(0)) #shifted backwards (to left)
	#data_window[-1] = new_data_point #Replace with new data point
	previous_window_data = data_window.pop(0) #Pop off first element in window and store
	data_window.append(new_data) #Append new data point at end of window
	
	previous_window_data_ffill = data_ffill_window.pop(0) #Pop off first element in window and store
	data_ffill_window.append(new_data_ffill) #Append new data point at end of window
	
	previous_window_hour = hour_window.pop(0) #Pop off first element in window and store
	hour_window.append(new_hour) #Append new data point at end of window
	#print(f"Replacing {previous_window_last_value} with {new_data}")
	
	window = []	   
		
	#encoded_cyclical_features= ['cos_hour','sin_hour']
	for i,(data,data_ffill,hour) in enumerate(zip(data_window,data_ffill_window,hour_window)):
		hour_norm = (2*math.pi*hour)/23.0 # We min-max normalize x values to match with the 0-2π cycle
		features = []
		for input_feature in input_features:
			if data == 0.0:
				if input_feature == "load_value_corrupted":
					features.append(data)
				elif input_feature == "load_value_corrupted_ffill":
					features.append(data_ffill)
				#elif input_feature == "load_value_corrupted_ffill":
				#	 if i>=1:
				#		 features.append(data_window[i-1])
				#	 else:
				#		 features.append(previous_window_data)
				elif input_feature == "corruption_encoding":
					features.append(1.0)
				elif input_feature == "cos_hour":
					features.append(np.cos(hour_norm))
				elif input_feature == "sin_hour":
					features.append(np.sin(hour_norm))
			else:
				if input_feature == "load_value_corrupted":
					features.append(data)
				elif input_feature == "load_value_corrupted_ffill":
					features.append(data_ffill)
				elif input_feature == "corruption_encoding":
					features.append(0.0)
				elif input_feature == "cos_hour":
					features.append(np.cos(hour_norm))
				elif input_feature == "sin_hour":
					features.append(np.sin(hour_norm))
		window.append(features)		  
	
	return np.array(window)

def create_streaming_data(new_data,new_data_ffill,new_hour,data_window,data_ffill_window,hour_window,load_block_length,input_features):
	"""Returns a window that can be used as input to data imputation model that simulates a moving window"""
	
	#print(input_features)
	assert len(data_window)== len(hour_window) == load_block_length, f"Expected data window:{data_window} to have {load_block_length} elements"
	previous_window_data = data_window.pop(0) #Pop off first element in window and store
	data_window.append(new_data) #Append new data point at end of window
	
	previous_window_data_ffill = data_ffill_window.pop(0) #Pop off first element in window and store
	data_ffill_window.append(new_data_ffill) #Append new data point at end of window
	
	previous_window_hour = hour_window.pop(0) #Pop off first element in window and store
	hour_window.append(new_hour) #Append new data point at end of window
	#print(f"Pushing out data:{previous_window_data:2f} at hour:{previous_window_hour}...appending data:{new_data:2f} at hour:{new_hour}")
	window = []	   
		
	#encoded_cyclical_features= ['cos_hour','sin_hour']
	for i,(data,data_ffill,hour) in enumerate(zip(data_window,data_ffill_window,hour_window)):
		hour_norm = (2*math.pi*hour)/23.0 # We min-max normalize x values to match with the 0-2π cycle
		features = []
		for input_feature in input_features:
			#print(i,input_feature)
			if data == 0.0:
				if input_feature == "load_value":
					features.append(data)
				elif input_feature == "load_value_corrupted":
					features.append(data)
				elif input_feature == "load_value_corrupted_ffill":
					features.append(data_ffill)				   
				elif input_feature == "corruption_encoding":
					features.append(1.0)
				elif input_feature == "cos_hour":
					features.append(np.cos(hour_norm))
				elif input_feature == "sin_hour":
					features.append(np.sin(hour_norm))
			else:
				if input_feature == "load_value":
					features.append(data)
				elif input_feature == "load_value_corrupted":
					features.append(data)
				elif input_feature == "load_value_corrupted_ffill":
					features.append(data_ffill)
				elif input_feature == "corruption_encoding":
					features.append(0.0)
				elif input_feature == "cos_hour":
					features.append(np.cos(hour_norm))
				elif input_feature == "sin_hour":
					features.append(np.sin(hour_norm))
		window.append(features)		  
	#print(window)
	return np.array(window)

def evaluate_dataimputation_on_streaming_data_multi_nodes(df_eval,autoencoder,monitored_nodes,load_block_length,input_features,timeinterval_mins=15):
	"""Loop through monitored nodes"""
	
	print(f"Following {len(monitored_nodes)} nodes will be observed:{monitored_nodes}")

	node_data_dict = {node_id:{"data_raw_window":[round(df_eval.loc[(df_eval["node_id"]==node_id),"load_value"].mean(),1)]*load_block_length,\
							   "data_ffill_window":[round(df_eval.loc[(df_eval["node_id"]==node_id),"load_value"].mean(),1)]*load_block_length,\
							   "data_actual_window":[round(df_eval.loc[(df_eval["node_id"]==node_id),"load_value"].mean(),1)]*load_block_length,\
							   "hour_window":[(df_eval.loc[0,"datetime"] - DateOffset(minutes=int(timeinterval_mins*(load_block_length-i)))).hour for i in range(load_block_length)],\
							   "timestamp_window":[(df_eval.loc[0,"datetime"] - DateOffset(minutes=int(timeinterval_mins*(load_block_length-i)))) for i in range(load_block_length)],\
							   "mse_di_model":[],"mse_ffill_model":[]}
	
				 for node_id in list(set(df_eval["node_id"]))}
	
	for i,timestamp in enumerate(df_eval["datetime"].unique()):
		print(f"Moving window:{i} starting at timestamp:{pd.Timestamp(timestamp)}")
		for monitored_node in monitored_nodes:
			df_eval_selected_node = df_eval.loc[(df_eval["node_id"]==monitored_node) & (df_eval["datetime"]==timestamp)]# ,["datetime","load_value","load_value_corrupted","load_value_corrupted_ffill"]]
			df_eval_selected_node = df_eval_selected_node.reset_index(drop=True)
			streaming_data_dict = {"timestamp":df_eval_selected_node.loc[0,"datetime"],"hour":df_eval_selected_node.loc[0,"datetime"].hour,
								   "data_raw":df_eval_selected_node.loc[0,"load_value_corrupted"],
								   "data_ffill":df_eval_selected_node.loc[0,"load_value_corrupted_ffill"],"data_actual":df_eval_selected_node.loc[0,"load_value"]}
			
			
			update_window_and_impute(streaming_data_dict,autoencoder,monitored_node,load_block_length,node_data_dict,i,input_features)
	
	for monitored_node in monitored_nodes:
		print(f"Node:{monitored_node} - DAE error:{sum(node_data_dict[monitored_node]['mse_di_model'])/len(node_data_dict[monitored_node]['mse_di_model']):.2f},ffill error:{sum(node_data_dict[monitored_node]['mse_ffill_model'])/len(node_data_dict[monitored_node]['mse_ffill_model']):.2f}")

def update_window_and_impute(streaming_data_dict,autoencoder,selected_node,load_block_length,node_data_dict,window_id,input_features):
	"""Evaluate on a datastream with missing values at time stamps"""
	
	window = create_streaming_data(streaming_data_dict["data_raw"],streaming_data_dict["data_ffill"],streaming_data_dict["hour"],
								   node_data_dict[selected_node]["data_raw_window"],node_data_dict[selected_node]["data_ffill_window"],node_data_dict[selected_node]["hour_window"],
								   load_block_length,input_features)
	
	if "data_actual_window" in node_data_dict[selected_node]:
		node_data_dict[selected_node]["data_actual_window"].append(node_data_dict[selected_node]["data_actual_window"].pop(0)) #shifted backwards (to left)
		node_data_dict[selected_node]["data_actual_window"][-1] =  streaming_data_dict["data_actual"] #Replace with new data point
		
	node_data_dict[selected_node]["timestamp_window"].append(node_data_dict[selected_node]["timestamp_window"].pop(0)) #shifted backwards (to left)
	node_data_dict[selected_node]["timestamp_window"][-1] =	 streaming_data_dict["timestamp"] #Replace with new data point
	
	
	preprocessing_output_dict = {timestamp:{"AE":node_data_dict[selected_node]["data_raw_window"][i],"ffill":node_data_dict[selected_node]["data_ffill_window"][i]} for i,timestamp in enumerate(node_data_dict[selected_node]["timestamp_window"])}
		
	if 0.0 in node_data_dict[selected_node]["data_raw_window"]:
		missing_index = node_data_dict[selected_node]["data_raw_window"].index(0.0) #find index of missing value in window
		missing_timestamp = node_data_dict[selected_node]['timestamp_window'][missing_index]
		
		prediction = autoencoder.predict(np.expand_dims(window, axis=0),verbose=0) #Use autoencoder model to perform imputation
		ae_imputed_value = prediction.flatten()[missing_index]
		ffill_imputed_value = node_data_dict[selected_node]['data_ffill_window'][missing_index]
		
		print(f"Missing data at time stamp:{missing_timestamp}-node:{selected_node} in moving window:{window_id}- AE imputation:{ae_imputed_value:.2f},ffill:{ffill_imputed_value:.2f}")
		
		if "data_actual_window" in node_data_dict[selected_node]:
			node_data_dict[selected_node]["mse_di_model"].append((node_data_dict[selected_node]["data_actual_window"][missing_index] - ae_imputed_value)**2)
			node_data_dict[selected_node]["mse_ffill_model"].append((node_data_dict[selected_node]["data_actual_window"][missing_index] - ffill_imputed_value)**2)
		
		preprocessing_output_dict[missing_timestamp]["AE"] = ae_imputed_value
	
	return preprocessing_output_dict
		

def impute_on_streaming_data_legacy(prediction_model,data_window,previous_window_last_value):
	"""Impute missing values on one instance of a window"""
	#data_window shape should be (1,window_size,number_of_features) so that predict method does not throw an error
	
	if 0.0 in data_window[0,:,0]: #Check if data is missing in window
		indexes_with_missing_values = np.where(data_window[0,:,0] == 0.0)
		#print(indexes_with_missing_values)
		predictions_model = prediction_model.predict(data_window).flatten() #Only use DAE predictions if data is missing in the window
		print(f"DAE predictions:{predictions_model}")
		predictions_ffill = []
		for i,value in enumerate(data_window[0,:,0]):
			if value == 0.0:
				if i==0:
					predictions_ffill.append(previous_window_last_value)
				else:
					predictions_ffill.append(data_window[0,i-1,0]) #Use forward fill if data is missing in the window
			else:
				predictions_ffill.append(value)
		print(f"FFILL predictions:{predictions_ffill}")
	else:
		predictions_model = data_window[0,:,0] #Use original data window if no value is missing
		predictions_ffill = data_window[0,:,0] #Use original data window if no value is missing
	
	return predictions_model,predictions_ffill

def get_ffill_imputation_error_from_df(df):
	
	mse_ffill = ((df[df["data_quality"]=="corrupted"]["load_value"]-df[df["data_quality"]=="corrupted"]["load_value_corrupted_ffill"])**2).mean()
	
	return mse_fill

def compare_performance_moving_window(df,predictions,window_size,measurement_column,n_windows = 3000000,alternate_predictions={}):
	"""Compare performance for moving window"""
	
	input_features=[f"{measurement_column}", "corruption_encoding",f"{measurement_column}_corrupted_ffill"]
		
	dataset_moving_window,_ = get_input_target_dataset(df,window_size,input_features,target_feature=None,batch_size=None,use_moving_window=True)
	
	print(f"Taking {n_windows} element from dataset with cardinality:{dataset_moving_window.cardinality().numpy()} and converting to Numpy array...")
	array_moving_window = np.array(list(dataset_moving_window.take(n_windows).as_numpy_iterator()))
	print(f"Array shape from dataset:{array_moving_window.shape}")
	array_moving_window = np.array(array_moving_window).reshape(-1,len(input_features))
	print(f"Array shape after reshape:{array_moving_window[:,0].shape}")
	
	df_comparison = pd.DataFrame()
	df_comparison[f"{measurement_column}_actual"] =  array_moving_window[:,0]
	df_comparison["corruption_encoding"] =	array_moving_window[:,1]
	df_comparison[f"{measurement_column}_ffill"] =	 array_moving_window[:,2]
	df_comparison[f"{measurement_column}_DAE"] =  predictions[0:n_windows].reshape(-1,1)[:,-1]
	for alternate in alternate_predictions.keys(): #Add alterante predictions
		print(f"Adding alternate prediction:{alternate}")
		df_comparison[f"{measurement_column}_{alternate}"] = alternate_predictions[alternate][0:n_windows].reshape(-1,1)[:,-1] 
	
	df_comparison["prediction_ffill_AE"] = abs(df_comparison[f"{measurement_column}_actual"]-df_comparison[f"{measurement_column}_ffill"])
	df_comparison["prediction_DAE_AE"] = abs(df_comparison[f"{measurement_column}_actual"]-df_comparison[f"{measurement_column}_DAE"])
	for alternate in alternate_predictions.keys(): #Add alterante predictions
		df_comparison[f"prediction_{alternate}_AE"] = abs(df_comparison[f"{measurement_column}_actual"]-df_comparison[f"{measurement_column}_{alternate}"])
	
	df_comparison["prediction_ffill_SE"] = np.square(df_comparison[f"{measurement_column}_actual"].values- df_comparison[f"{measurement_column}_ffill"].values)
	df_comparison["prediction_DAE_SE"] = np.square(df_comparison[f"{measurement_column}_actual"].values- df_comparison[f"{measurement_column}_DAE"].values)
	for alternate in alternate_predictions.keys(): #Add alterante predictions
		df_comparison[f"prediction_{alternate}_SE"] = np.square(df_comparison[f"{measurement_column}_actual"].values- df_comparison[f"{measurement_column}_{alternate}"].values)
	
	print(f"Calculation imputation accuracy meterics for {len(df_comparison[df_comparison['corruption_encoding']==1.0])} missing values...")
	ffill_mae = df_comparison[df_comparison["corruption_encoding"]==1.0]["prediction_ffill_AE"].mean()
	DAE_mae = df_comparison[df_comparison["corruption_encoding"]==1.0]["prediction_DAE_AE"].mean()
		
	ffill_mse = df_comparison[df_comparison["corruption_encoding"]==1.0]["prediction_ffill_SE"].mean()
	DAE_mse = df_comparison[df_comparison["corruption_encoding"]==1.0]["prediction_DAE_SE"].mean()
	
	alternate_metrics ={}	 
	for alternate in alternate_predictions.keys(): #Add alterante predictions
		alternate_metrics.update({f"{alternate}_mae":df_comparison[df_comparison["corruption_encoding"]==1.0][f"prediction_{alternate}_AE"].mean()})
		alternate_metrics.update({f"{alternate}_mse":df_comparison[df_comparison["corruption_encoding"]==1.0][f"prediction_{alternate}_SE"].mean()})
		alternate_metrics.update({f"{alternate}_rmse":alternate_metrics[f"{alternate}_mse"]**0.5})
	
	ffill_rmse = ffill_mse**0.5
	DAE_rmse = DAE_mse**0.5
	
	print(f"MAE - ffill:{ffill_mae:.3f},DAE:{DAE_mae:.3f}")
	print(f"MSE - ffill:{ffill_mse:.3f},DAE:{DAE_mse:.3f}")
	print(f"RMSE - ffill:{ffill_rmse:.3f},DAE:{DAE_rmse:.3f}")
	for alternate in alternate_predictions.keys(): #Add alterante predictions
		print(f"{alternate} - MAE:{alternate_metrics[f'{alternate}_mae']:.3f},MSE::{alternate_metrics[f'{alternate}_mse']:.3f},RMSE::{alternate_metrics[f'{alternate}_rmse']:.3f}")
	
	print(f"Relative improvement in MAE - ffill:{((ffill_mae-DAE_mae)/ffill_mae)*100:.2f}")
	print(f"Relative improvement in RMSE - ffill:{((ffill_rmse-DAE_rmse)/ffill_rmse)*100:.2f}")
	for alternate in alternate_predictions.keys(): #Add alterante predictions
		print(f"Relative improvement in MAE - {alternate}:{((alternate_metrics[f'{alternate}_mae']-DAE_mae)/alternate_metrics[f'{alternate}_mae'])*100:.2f}")
		print(f"Relative improvement in RMSE - {alternate}:{((alternate_metrics[f'{alternate}_rmse']-DAE_rmse)/alternate_metrics[f'{alternate}_rmse'])*100:.2f}")

	print(df_comparison[df_comparison["corruption_encoding"]==1.0].head(10))
	print(df_comparison[df_comparison["corruption_encoding"]==1.0].describe().loc["mean",:])
	
	return df_comparison

def predict_from_time_series_files(time_series_files,prediction_model_file,load_block_length,input_features,target_feature):
	"""Make data imputation on time series files after corrupting it"""
	
	df,input_target = get_dataset_from_csv(time_series_files,load_type_selected,load_block_length,selected_month,input_features,target_feature,show_details=False,df_type="eval",corrupted_fraction=0.01,use_moving_window=False)
		
	print(f"Input dataset shape:{input_target.take(1).as_numpy_iterator().next()[0].shape}")
	print(f"Target dataset shape:{input_target.take(1).as_numpy_iterator().next()[1].shape}")
	
	check_moving_window(input_target,df,load_block_length,input_features)
	input_target =input_target.batch(128)	 
	
	predictions = load_evaluate_predict(prediction_model_file,input_target=input_target)
	compare_performance_moving_window(df,predictions, load_block_length,n_windows = 1400000)
	
	return predictions
