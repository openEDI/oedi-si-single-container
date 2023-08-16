"""'
Created on Wed December 5 10:00:00 2022
@author: Siby Plathottam
"""

import os
import sys
import time
import math
import pickle
import random
from typing import List, Set, Dict, Tuple, Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from pandas.tseries.offsets import DateOffset
from numpy.random import default_rng

def compare_performance_moving_window(df,predictions,load_block_length,input_features,target_features, n_windows = 3000000,alternate_predictions={}):
	"""Compare performance for moving window"""
	
	dataset_input =tf.keras.utils.timeseries_dataset_from_array(
							data=df[input_features].values,
							targets = None,
							sequence_length =load_block_length,
							sequence_stride=1,
							sampling_rate=1,
							batch_size=None,
							shuffle=False,
							seed=None,
							start_index=None,
							end_index=None)
	
	dataset_target =tf.keras.utils.timeseries_dataset_from_array(
							data=df[target_features].values,
							targets = None,
							sequence_length =load_block_length ,
							sequence_stride=1,
							sampling_rate=1,
							batch_size=None,
							shuffle=False,
							seed=None,
							start_index=None,
							end_index=None)
	
	print(f"Taking {n_windows} element from dataset with cardinality:{dataset_input.cardinality().numpy()} and converting to Numpy array...")
	input_array_moving_window = np.array(list(dataset_input.take(n_windows).as_numpy_iterator()))
	print(f"Array shape from dataset:{input_array_moving_window.shape}")
	input_array_moving_window = np.array(input_array_moving_window).reshape(-1,len(input_features))
	print(f"Array shape after reshape:{input_array_moving_window[:,0].shape}")
	
	print(f"Taking {n_windows} element from target dataset with cardinality:{dataset_target.cardinality().numpy()} and converting to Numpy array...")
	target_array_moving_window = np.array(list(dataset_target.take(n_windows).as_numpy_iterator()))
	print(f"Target array shape from dataset:{target_array_moving_window.shape}")
	target_array_moving_window = np.array(target_array_moving_window).reshape(-1,len(target_features))
	print(f"Target array shape after reshape:{target_array_moving_window[:,0].shape}")
	
	df_comparison = pd.DataFrame()
	df_comparison["solar_actual"] =	 target_array_moving_window[:,0]
	df_comparison["solar_predicted"] =	predictions[0:n_windows].reshape(-1,1)[:,-1]
	
	for alternate in alternate_predictions.keys(): #Add alterante predictions
		print(f"Adding alternate prediction:{alternate}")
		df_comparison[f"load_value_{alternate}"] =	alternate_predictions[alternate][0:n_windows].reshape(-1,1)[:,-1] 
	
	df_comparison["prediction_AE"] = abs(df_comparison["solar_actual"]-df_comparison["solar_predicted"])
	for alternate in alternate_predictions.keys(): #Add alterante predictions
		df_comparison[f"prediction_{alternate}_AE"] = abs(df_comparison["solar_actual"]-df_comparison[f"solar_predicted_{alternate}"])
	
	df_comparison["prediction_SE"] = np.square(df_comparison["solar_actual"].values- df_comparison["solar_predicted"].values)
	for alternate in alternate_predictions.keys(): #Add alterante predictions
		df_comparison[f"prediction_{alternate}_SE"] = np.square(df_comparison["solar_actual"].values- df_comparison[f"solar_predicted_{alternate}"].values)
	
	prediction_mae = df_comparison["prediction_AE"].mean()
	prediction_mse = df_comparison["prediction_SE"].mean()
	
	alternate_metrics ={}	 
	for alternate in alternate_predictions.keys(): #Add alterante predictions
		alternate_metrics.update({f"{alternate}_mae":df_comparison[f"prediction_{alternate}_AE"].mean()})
		alternate_metrics.update({f"{alternate}_mse":df_comparison[f"prediction_{alternate}_SE"].mean()})
		alternate_metrics.update({f"{alternate}_rmse":alternate_metrics[f"{alternate}_mse"]**0.5})
	
	prediction_rmse = prediction_mse**0.5
	
	print(f"MAE -model:{prediction_mae:.3f}")
	print(f"MSE - model:{prediction_mse:.3f}")
	print(f"RMSE - model:{prediction_rmse:.3f}")
	for alternate in alternate_predictions.keys(): #Add alterante predictions
		print(f"{alternate} - MAE:{alternate_metrics[f'{alternate}_mae']:.3f},MSE::{alternate_metrics[f'{alternate}_mse']:.3f},RMSE::{alternate_metrics[f'{alternate}_rmse']:.3f}")
	
	for alternate in alternate_predictions.keys(): #Add alterante predictions
		print(f"Relative improvement in MAE - {alternate}:{((alternate_metrics[f'{alternate}_mae']-DAE_mae)/alternate_metrics[f'{alternate}_mae'])*100:.2f}")
		print(f"Relative improvement in RMSE - {alternate}:{((alternate_metrics[f'{alternate}_rmse']-DAE_rmse)/alternate_metrics[f'{alternate}_rmse'])*100:.2f}")

	return df_comparison

def evaluate_disaggregation_on_streaming_data_multi_nodes(df_eval,model,monitored_nodes,load_block_length,input_features,timeinterval_mins=15):
	"""Loop through monitored nodes"""
	
	print(f"Following {len(monitored_nodes)} nodes will be observed:{monitored_nodes}")

	node_data_dict = {node_id:{"data_raw_window":[round(df_eval.loc[(df_eval["node_id"]==node_id),"net_load"].mean(),1)]*load_block_length,\
							   "data_actual_window":[round(df_eval.loc[(df_eval["node_id"]==node_id),"solar_power"].mean(),1)]*load_block_length,\
							   "hour_window":[(df_eval.loc[0,"datetime"] - DateOffset(minutes=int(timeinterval_mins*(load_block_length-i)))).hour for i in range(load_block_length)],\
							   "day_of_week_window":[(df_eval.loc[0,"datetime"] - DateOffset(minutes=int(timeinterval_mins*(load_block_length-i)))).hour for i in range(load_block_length)],\
							   "timestamp_window":[(df_eval.loc[0,"datetime"] - DateOffset(minutes=int(timeinterval_mins*(load_block_length-i)))) for i in range(load_block_length)],\
							   "mse_disaggregation_model":[]}
	
				 for node_id in list(set(df_eval["node_id"]))}
	
	for i,timestamp in enumerate(df_eval["datetime"].unique()):
		print(f"Moving window:{i} starting at timestamp:{pd.Timestamp(timestamp)}")
		for monitored_node in monitored_nodes:
			df_eval_selected_node = df_eval.loc[(df_eval["node_id"]==monitored_node) & (df_eval["datetime"]==timestamp)]# ,["datetime","load_value","load_value_corrupted","load_value_corrupted_ffill"]]
			df_eval_selected_node = df_eval_selected_node.reset_index(drop=True)
			streaming_data_dict = {"timestamp":df_eval_selected_node.loc[0,"datetime"],"hour":df_eval_selected_node.loc[0,"datetime"].hour,"day_of_week":df_eval_selected_node.loc[0,"datetime"].dayofweek,
								   "data_raw":df_eval_selected_node.loc[0,"net_load"],
								   "data_actual":df_eval_selected_node.loc[0,"solar_power"]}
			
			
			update_window_and_disaggregate(streaming_data_dict,model,monitored_node,load_block_length,node_data_dict,i,input_features)
	
	for monitored_node in monitored_nodes:
		print(f"Node:{monitored_node} - disaggregation error:{sum(node_data_dict[monitored_node]['mse_disaggregation_model'])/len(node_data_dict[monitored_node]['mse_disaggregation_model']):.2f}")

def update_window_and_disaggregate(streaming_data_dict,model,selected_node,load_block_length,node_data_dict,window_id,input_features):
	"""Evaluate on a datastream with missing values at time stamps"""
	
	window = create_streaming_data_disaggregation(streaming_data_dict["data_raw"],streaming_data_dict,node_data_dict[selected_node]["data_raw_window"],node_data_dict[selected_node],load_block_length,input_features)
	
	if "data_actual_window" in node_data_dict[selected_node]:
		node_data_dict[selected_node]["data_actual_window"].append(node_data_dict[selected_node]["data_actual_window"].pop(0)) #shifted backwards (to left)
		node_data_dict[selected_node]["data_actual_window"][-1] =  streaming_data_dict["data_actual"] #Replace with new data point
		
	node_data_dict[selected_node]["timestamp_window"].append(node_data_dict[selected_node]["timestamp_window"].pop(0)) #shifted backwards (to left)
	node_data_dict[selected_node]["timestamp_window"][-1] =	 streaming_data_dict["timestamp"] #Replace with new data point
	
	
	disaggregation_output_dict = {timestamp:{"prediction":node_data_dict[selected_node]["data_raw_window"][i]} for i,timestamp in enumerate(node_data_dict[selected_node]["timestamp_window"])}
	
	prediction = model.predict(np.expand_dims(window, axis=0),verbose=0) #Use model to perform prediction
	for i,data_raw in enumerate(node_data_dict[selected_node]["data_raw_window"]):
		prediction_index = i #find index of missing value in window
		prediction_timestamp = node_data_dict[selected_node]['timestamp_window'][prediction_index]
		predicted_value = prediction.flatten()[prediction_index]
		
		#print(f"Time stamp:{prediction_timestamp}-node:{selected_node} in moving window:{window_id}- Net load:{data_raw:.2f},Solar prediction:{predicted_value:.2f}")
		
		if "data_actual_window" in node_data_dict[selected_node]:
			node_data_dict[selected_node]["mse_disaggregation_model"].append((node_data_dict[selected_node]["data_actual_window"][prediction_index] - predicted_value)**2)
			
		disaggregation_output_dict[prediction_timestamp]["prediction"] = predicted_value
	
	return disaggregation_output_dict

def create_streaming_data_disaggregation(new_data,new_time_dict,data_window,time_window_dict,load_block_length,input_features):
	"""Returns a window that can be used as input to data imputation model that simulates a moving window"""
	
	#print(input_features)	  
	assert len(data_window)== load_block_length, f"Expected data window:{data_window} to have {load_block_length} elements"
	previous_window_data = data_window.pop(0) #Pop off first element in window and store
	data_window.append(new_data) #Append new data point at end of window
	#print(f"Pushing out data:{previous_window_data:2f}..appending data:{new_data:2f}")
	
	if "hour" in new_time_dict:
		assert len(data_window)== len(time_window_dict["hour_window"]), f"Expected hour window:{time_window_dict['hour_window']} to have {load_block_length} elements"
		previous_window_hour = time_window_dict["hour_window"].pop(0) #Pop off first element in window and store
		time_window_dict["hour_window"].append(new_time_dict["hour"]) #Append new data point at end of window
		
	if "day_of_week	 " in new_time_dict:
		assert len(data_window)== len(time_window_dict["day_of_week	 _window"]), f"Expected day of week window:{time_window_dict['day_of_week  _window']} to have {load_block_length} elements"
		previous_window_day = time_window_dict["day_window"].pop(0) #Pop off first element in window and store
		time_window_dict["day_of_week_window"].append(new_time_dict["day_of_week"]) #Append new data point at end of window
			   
	window = []	   
	
	for i,data in enumerate(data_window):
		
		if "hour_window" in time_window_dict:
			hour = time_window_dict["hour_window"][i]
			hour_norm = (2*math.pi*hour)/23.0 # We min-max normalize x values to match with the 0-2π cycle
		if "day_of_week_window" in time_window_dict:
			day_of_week = time_window_dict["day_of_week_window"][i]
			day_of_week_norm = (2*math.pi*day_of_week)/ 6.0	 # We normalize x values to match with the 0-2π cycle
		features = []
		for input_feature in input_features:
			#print(i,input_feature)
			if input_feature == "net_load":
				features.append(data)
			elif input_feature == "cos_hour":
				features.append(np.cos(hour_norm))
			elif input_feature == "sin_hour":
				features.append(np.sin(hour_norm))
			elif input_feature == "cos_day_of_week":
				features.append(np.cos(day_of_week_norm))
			elif input_feature == "sin_day_of_week":
				features.append(np.sin(day_of_week_norm))
			else:
				raise ValueError(f"{input_feature} is an invalid feature!")
		window.append(features)
	#print(window)
	return np.array(window)  
		
		
