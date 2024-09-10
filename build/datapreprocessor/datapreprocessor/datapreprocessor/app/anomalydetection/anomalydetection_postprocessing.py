"""'
Created on Nov 15 10:00:00 2022
@author: Siby Plathottam
"""

import math
from typing import List, Set, Dict, Tuple, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset

def evaluate_anomalydetection_on_streaming_data_multi_nodes(df_eval,autoencoder,monitored_nodes,load_block_length,input_features,timeinterval_mins=15,reconstruction_error_threshold=1e-2):
	"""Loop through monitored nodes"""
	
	print(f"Following {len(monitored_nodes)} nodes will be observed:{monitored_nodes}")
	print(f"Reconstruction error threshold for anomaly:{reconstruction_error_threshold}")

	node_data_dict = {node_id:{"data_raw_window":[round(df_eval.loc[(df_eval["node_id"]==node_id),"load_value"].mean(),1)]*load_block_length,\
							   "hour_window":[(df_eval.loc[0,"datetime"] - DateOffset(minutes=int(timeinterval_mins*(load_block_length-i)))).hour for i in range(load_block_length)],\
							   "timestamp_window":[(df_eval.loc[0,"datetime"] - DateOffset(minutes=int(timeinterval_mins*(load_block_length-i)))) for i in range(load_block_length)],\
							   "anomaly_detected_count":0,"anomaly_detected_timestamps":[]}
	
				 for node_id in list(set(df_eval["node_id"]))}
	
	for i,timestamp in enumerate(df_eval["datetime"].unique()):
		print(f"Moving window:{i} starting at timestamp:{pd.Timestamp(timestamp)}")
		for monitored_node in monitored_nodes:
			df_eval_selected_node = df_eval.loc[(df_eval["node_id"]==monitored_node) & (df_eval["datetime"]==timestamp)]
			df_eval_selected_node = df_eval_selected_node.reset_index(drop=True)
			streaming_data_dict = {"timestamp":df_eval_selected_node.loc[0,"datetime"],"hour":df_eval_selected_node.loc[0,"datetime"].hour,
								   "day_of_week":df_eval_selected_node.loc[0,"datetime"].day_of_week,
								   "data_raw":df_eval_selected_node.loc[0,"load_value_anomaly"]}
			node_data_dict = update_window_and_detectanomaly(streaming_data_dict,autoencoder,monitored_node,load_block_length,node_data_dict,i,input_features,reconstruction_error_threshold)
	
	for monitored_node in monitored_nodes:
		print(f"Node:{monitored_node} - Aggregate anomalies detected:{node_data_dict[monitored_node]['anomaly_detected_count']},\
										Anomaly timestamps:{len(set(node_data_dict[monitored_node]['anomaly_detected_timestamps']))}")
		
	plt.figure(figsize=(10,10))
	for monitored_node in monitored_nodes:
		plt.plot(df_eval.loc[(df_eval["node_id"]==monitored_node),"datetime"],df_eval.loc[(df_eval["node_id"]==monitored_node),"load_value_anomaly"], label=monitored_node)
		for anomaly_detected_timestamp in node_data_dict[monitored_node]['anomaly_detected_timestamps']:
			plt.axvspan(anomaly_detected_timestamp,anomaly_detected_timestamp + DateOffset(minutes=timeinterval_mins),color='green', alpha=0.1)
			#plt.axvspan(anomaly_detected_timestamp,anomaly_detected_timestamp,color='green', alpha=0.1)

	plt.legend()
	plt.xlabel('Time stamps')
	plt.ylabel('Load (kWh)')
	plt.title(f'window size:{load_block_length}')
	plt.savefig("anomaly_detection_time_series.png")
	plt.show()	
		
	return node_data_dict

def create_streaming_data_anomalydetection(new_data,new_time_dict,data_window,time_window_dict,load_block_length,input_features):
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
			if input_feature == "load_value":
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


def update_window_and_detectanomaly(streaming_data_dict,autoencoder,selected_node,load_block_length,node_data_dict,window_id,input_features,reconstruction_error_threshold,show_details=True):
	"""Evaluate on a datastream for anomalous values at time stamps"""
	
	window = create_streaming_data_anomalydetection(streaming_data_dict["data_raw"],streaming_data_dict,node_data_dict[selected_node]["data_raw_window"],node_data_dict[selected_node],load_block_length,input_features)
	
	node_data_dict[selected_node]["timestamp_window"].append(node_data_dict[selected_node]["timestamp_window"].pop(0)) #shifted backwards (to left)
	node_data_dict[selected_node]["timestamp_window"][-1] =	 streaming_data_dict["timestamp"] #Replace with new data point
	window = np.expand_dims(window, axis=0)
	reconstruction = autoencoder.predict(window,verbose=0).flatten() #Use model to perform reconstruction
	original = window[:,:,0].flatten()
	
	reconstruction_error = ((original-reconstruction)**2).flatten()	
	max_reconstruction_error = reconstruction_error[reconstruction_error>reconstruction_error_threshold].round(2)
	
	
	node_data_dict[selected_node].update({streaming_data_dict["timestamp"]:{"data_raw":streaming_data_dict["data_raw"]}})
	node_data_dict[selected_node][streaming_data_dict["timestamp"]].update({"anomaly":False}) #Set false if new time stamp does not have anomaly
	node_data_dict[selected_node][streaming_data_dict["timestamp"]].update({"reconstruction_error":reconstruction_error[-1]}) #SUpdate reconstruction error for latest timestamp
	if len(max_reconstruction_error) > 0:
		max_reconstruction_error_index = np.argwhere(reconstruction_error>reconstruction_error_threshold).flatten() #np.nonzero(reconstruction_error[reconstruction_error>reconstruction_error_threshold])[0]
		anomaly_timestamps = [node_data_dict[selected_node]['timestamp_window'][i] for i in max_reconstruction_error_index]
		anomaly_data = [round(node_data_dict[selected_node]['data_raw_window'][i],2) for i in max_reconstruction_error_index]
		if show_details:
			print(f"Data window:{node_data_dict[selected_node]['data_raw_window']},Reconstruction error:{reconstruction_error.round(2)}")
			print(f"Anomaly detected at time stamp:{anomaly_timestamps}-node:{selected_node} in moving window:{window_id},index:{max_reconstruction_error_index}")
			print(f"Anomalous data:{anomaly_data} - AE reconstruction error:{max_reconstruction_error}")

		node_data_dict[selected_node]['anomaly_detected_count'] = node_data_dict[selected_node]['anomaly_detected_count'] + len(anomaly_data)
		node_data_dict[selected_node]['anomaly_detected_timestamps'].extend(anomaly_timestamps)
                
		if streaming_data_dict["timestamp"] in anomaly_timestamps:
			node_data_dict[selected_node][streaming_data_dict["timestamp"]]["anomaly"] = True #Set true if new time stamp has anomaly
			print(f"Anomaly at {streaming_data_dict['timestamp']} due to data:{node_data_dict[selected_node][streaming_data_dict['timestamp']]['data_raw']:.2f}")
    
	return node_data_dict

def update_window_and_detectanomaly_legacy(df_row,autoencoder,selected_node,load_block_length,node_data_dict,window_id,input_features,reconstruction_error_threshold):
	"""Evaluate on a datastream with missing values at time stamps"""
		
	new_timestamp = df_row.loc[0,"datetime"]
	new_hour = df_row.loc[0,"datetime"].hour	
	new_data_raw = df_row.loc[0,"load_value_anomaly"]
	
	window = create_streaming_data(new_data_raw,0,new_hour,
								   node_data_dict[selected_node]["data_raw_window"],[0]*load_block_length,node_data_dict[selected_node]["hour_window"],
								   load_block_length,input_features)
	
	node_data_dict[selected_node]["timestamp_window"].append(node_data_dict[selected_node]["timestamp_window"].pop(0)) #shifted backwards (to left)
	node_data_dict[selected_node]["timestamp_window"][-1] =	 new_timestamp #Replace with new data point
		
	#print(f"Window ID:{window_id} - {node_data_dict[selected_node]['data_raw_window']}")
	window = np.expand_dims(window, axis=0)
	reconstruction = autoencoder.predict(window,verbose=0) #Use autoencoder model to perform imputation
	reconstruction = reconstruction.flatten()
	original = window[:,:,0].flatten()
	reconstruction_error = ((original-reconstruction)**2).flatten()	  
	
	max_reconstruction_error = reconstruction_error[reconstruction_error>reconstruction_error_threshold].round(2)
	if len(max_reconstruction_error) > 0:
		max_reconstruction_error_index = np.nonzero(reconstruction_error[reconstruction_error>reconstruction_error_threshold])[0]
		
		anomaly_timestamps = [node_data_dict[selected_node]['timestamp_window'][i] for i in max_reconstruction_error_index]
		anomaly_data = [round(node_data_dict[selected_node]['data_raw_window'][i],2) for i in max_reconstruction_error_index]
		
		print(f"Anomaly detected at time stamp:{anomaly_timestamps}-node:{selected_node} in moving window:{window_id},index:{max_reconstruction_error_index}")
		print(f"Anomalous data:{anomaly_data} - AE error:{max_reconstruction_error}")
		print(f"Data window:{node_data_dict[selected_node]['data_raw_window']},Reconstruction error:{reconstruction_error.round(2)}")
		node_data_dict[selected_node]['anomaly_detected_count'] = node_data_dict[selected_node]['anomaly_detected_count'] + len(anomaly_data)
		node_data_dict[selected_node]['anomaly_detected_timestamps'].extend(anomaly_timestamps)
	
	return node_data_dict

def count_anomalies_in_df(df,anomaly_value_types,monitored_nodes):
	print(f"Following anomaly values:{anomaly_value_types} will be counted in following nodes:{monitored_nodes}")
	anomaly_value_count_dict = {}
	total_count = 0
	for anomaly_value_type in anomaly_value_types:
		anomaly_value_count_dict[f"{anomaly_value_type}"] = len(df[df["node_id"].isin(monitored_nodes) & (df["load_value_anomaly"]==anomaly_value_type)])        
		total_count += len(df[df["node_id"].isin(monitored_nodes) & (df["load_value_anomaly"]==anomaly_value_type)])
	anomaly_value_count_dict["total"] = total_count
	return anomaly_value_count_dict
