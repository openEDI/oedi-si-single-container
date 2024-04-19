"""
Created on Thursday Feb 26 15:00:00 2023
@author: Siby Plathottam
"""
import time
import pickle
import random

import tensorflow as tf

def df_to_input_target_dataset(df,load_block_length,input_features,target_feature,batch_size,use_prefetch,df_type = "train"):
	"""Get input target dataset that can be used by model.fit()"""
	
	input_dataset, target_dataset =	 get_input_target_dataset(df,load_block_length,input_features,target_feature,batch_size=None,use_moving_window=True)
	input_target = tf.data.Dataset.zip((input_dataset, target_dataset))
	print(f"First two elements in {df_type} dataset:{list(input_target.take(2).as_numpy_iterator())[0]}")

	check_moving_window(input_target,df,load_block_length,input_features)	
	cardinality = input_target.cardinality().numpy()
	
	print(f"TF {df_type} dataset Cardinality:{cardinality} - df size:{len(df)}")
	print(f"NAN in {df_type}:{df[input_features].isnull().values.any()}")#,NAN in Test:{df_test[input_features].isnull().values.any()}")
	
	if df_type == "train":
		if use_prefetch:
			input_target = input_target.cache().shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)			
		else:
			input_target = input_target.cache().shuffle(10000).batch(batch_size)			
	elif (df_type == "test") or (df_type == "eval"):
		input_target =input_target.batch(batch_size)
	else:
		raise ValueError(f"{df_type} is not a valid df type!")
	
	return input_target

def get_input_target_dataset(df,load_block_length,input_features,target_feature,batch_size=128,use_moving_window=False):
	"""Create TF datasets"""
		
	if use_moving_window:
		sequence_stride = 1
	else:
		sequence_stride = load_block_length #Uses a fixed window
	
	dataset_input =tf.keras.utils.timeseries_dataset_from_array(
							data=df[input_features].values,
							targets = None,
							sequence_length =load_block_length ,
							sequence_stride=sequence_stride,
							sampling_rate=1,
							batch_size=batch_size,
							shuffle=False,
							seed=None,
							start_index=None,
							end_index=None)
	  
	if target_feature:
		dataset_target =tf.keras.utils.timeseries_dataset_from_array(
							data=df[target_feature].values,
							targets = None,
							sequence_length =load_block_length ,
							sequence_stride=sequence_stride,
							sampling_rate=1,
							batch_size=batch_size,
							shuffle=False,
							seed=None,
							start_index=None,
							end_index=None)	
	else:
		dataset_target = None
	
	return dataset_input,dataset_target

def check_moving_window(dataset,df,load_block_length,input_features,n_samples = 10):
	"""Check difference between data set and dataframe after applying moving window"""
	
	difference_flag = False
	print(f"Checking moving window for window size {load_block_length} with input features:{input_features} on {n_samples} samples.")
	for i,input_target in enumerate(dataset.take(n_samples).as_numpy_iterator()): #Take n_samples from dataset and iterate
		difference = list(input_target)[0]-df[input_features][i:i+load_block_length].values
		if not difference.sum()==0.0:
			print(f"Difference detected at:{i} - difference:{difference}")
			difference_flag = True
	
	if not difference_flag:
		print("No difference detected!")
		
def tfdataset_to_pickle(input_target,pickle_file):
	"""Create pickle files with arrays that can be directly fed to data imputation model"""
	
	n_elements = input_target.cardinality().numpy()
	print(f"Converting {n_elements} elements into Numpy arrays")
	array_moving_window = list(input_target.as_numpy_iterator())
	features = []
	target = []
	for i in tqdm(range(n_elements)):
		features.append(array_moving_window[i][0])
		target.append(array_moving_window[i][1])
	features = np.array(features)
	target = np.array(target)	 

	print(f"Features shape:{features.shape}")
	print(f"Target shape:{target.shape}")

	data = {'features': features,'target': target}
	
	print(f"Saving to {pickle_file}")
	with open(pickle_file, 'wb') as handle:
		pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)	 
		
def windowed_dataset(df, window_size):
	dataset = tf.data.Dataset.from_tensor_slices(df)
	dataset = dataset.window(window_size, shift=1, drop_remainder=True)
	dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
	#dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
	#dataset = dataset.batch(batch_size).prefetch(1)
	return dataset

def windowed_dataset_v2(df, window_size,shuffle=False):
	dataset = tf.data.Dataset.from_tensor_slices(df)
	dataset = dataset.window(window_size, shift=1, drop_remainder=True)
	dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
	if not shuffle:
		dataset = dataset.map(lambda window: (window[:,1:], window[:,0]))
	else:
		dataset = dataset.shuffle(1000).map(lambda window: (window[:,1:], window[:,0]))
	#dataset = dataset.batch(batch_size).prefetch(1)
	return dataset

def benchmark_tfdataset(dataset, num_epochs=2):
	"""Benchmark a TF dataset"""
	
	start_time = time.perf_counter()
	for epoch_num in range(num_epochs):
		for sample in tqdm(dataset):
			# Performing a training step
			#time.sleep(0.001)
			pass
	print("Execution time:", time.perf_counter() - start_time)

def get_train_test_eval_nodes(node_dict,train_fraction=0.8,test_fraction=0.2):
	## Specify train and test nodes
	n_available_samples = len(node_dict.keys())
	train_test_fraction = train_fraction + test_fraction #1.0
	
	assert train_test_fraction <= 1.0, f"Train + test:{train_test_fraction} should be <= 1.0!"
	n_train_test_samples = int(n_available_samples*train_test_fraction) #80# # int(len(df_load_fraction)*train_test_fraction)
	
	n_train_samples = int(n_train_test_samples*train_fraction)
	n_test_samples = n_train_test_samples -	 n_train_samples
	n_eval_samples = n_available_samples - n_train_test_samples #len(df_load_fraction) - n_train_test_samples
	
	print(f"Total train samples:{n_train_samples}")
	print(f"Total test samples:{n_test_samples}")
	print(f"Total eval samples:{n_eval_samples}")
	available_nodes = list(node_dict.keys())
	train_nodes = random.sample(available_nodes,n_train_samples)
	test_nodes = list(set(available_nodes).symmetric_difference(set(train_nodes)))
	test_nodes = random.sample(test_nodes,n_test_samples)
	eval_nodes = list(set(available_nodes).symmetric_difference(set(train_nodes+test_nodes)))
	print(f"Train nodes:{len(train_nodes)},Test nodes:{len(test_nodes)},Eval nodes:{len(eval_nodes)}")
	
	return train_nodes,test_nodes,eval_nodes

def get_train_test_eval_timesteps(timestamps,train_fraction=0.8,test_fraction=0.2):
	"""Specify train and test timestamps"""
	n_available_samples = len(timestamps)
	train_test_fraction = train_fraction + test_fraction #1.0
	
	assert train_test_fraction <= 1.0, f"Train + test:{train_test_fraction} should be <= 1.0!"
	n_train_test_samples = int(n_available_samples*train_test_fraction) #80# # int(len(df_load_fraction)*train_test_fraction)
	
	n_train_samples = int(n_train_test_samples*train_fraction)
	n_test_samples = n_train_test_samples -	 n_train_samples
	n_eval_samples = n_available_samples - n_train_test_samples #len(df_load_fraction) - n_train_test_samples
	
	print(f"Total train samples:{n_train_samples}")
	print(f"Total test samples:{n_test_samples}")
	print(f"Total eval samples:{n_eval_samples}")
	available_timestamps = list(timestamps)
	train_timestamps = available_timestamps[0:n_train_samples]
	test_timestamps = available_timestamps[n_train_samples:n_train_samples+n_test_samples]
	eval_timestamps = available_timestamps[n_train_samples+n_test_samples:n_train_samples+n_test_samples+n_eval_samples]
	print(f"Train samples:{len(train_timestamps)},Test nodes:{len(test_timestamps)},Eval nodes:{len(eval_timestamps)}")
	
	return train_timestamps,test_timestamps,eval_timestamps
