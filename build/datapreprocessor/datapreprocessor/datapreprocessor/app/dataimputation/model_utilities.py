"""'
Created on Wed September 16 9:00:00 2022
@author: Siby Plathottam
"""

import pickle
import time
import os

import py7zr
import tensorflow as tf
from tensorflow.python.client import device_lib
from sklearn.impute import KNNImputer

def train_model(model,train_input_target,n_epochs,test_input_target,callbacks):
	tic = time.perf_counter()
	history = model.fit(train_input_target, epochs=n_epochs,shuffle=True,validation_data=test_input_target,callbacks=callbacks)
	toc = time.perf_counter()
	print(f"Training took:{toc-tic:.2f} s")
	num_threads = get_num_cpu_threads()
	print(f"Number of CPU threads used for training: {num_threads}")
	
	return model,history

def load_evaluate_predict(prediction_model_file,input_target=None,data_dict=None):
	"""Load model, evaluate, and predict using dataset or dictionary"""
	
	print(f"Predicting using model:{prediction_model_file}")
	prediction_model = tf.keras.models.load_model(prediction_model_file)
	if input_target is not None:
		model_performance_metrics = prediction_model.evaluate(input_target)
		predictions = prediction_model.predict(input_target)
	elif data_dict is not None:
		model_performance_metrics = prediction_model.evaluate(data_dict["features"],data_dict["target"])
		predictions = prediction_model.predict(data_dict["features"])
	else:
		raise ValueError("Either tf dataset or dict should be provided")
		
	print(f"Model performance metrics:{model_performance_metrics}")	   
	print(f"Predictions shape:{predictions.shape}")
	
	return predictions

def load_evaluate_predict_from_pickle(pickle_file,prediction_model_file):
	"""Load and predict from pickle file"""
	
	print(f"Reading {pickle_file}")
	with open(pickle_file, 'rb') as handle:
		data_dict = pickle.load(handle)
	
	predictions = load_evaluate_predict(prediction_model_file,data_dict=data_dict)
	
	return predictions

def get_knn_imputer(knn_array,n_neighbors=10):
	"""Get imputation from Scikit learn KNN imputation"""
	knn_imputer = KNNImputer(n_neighbors=n_neighbors)
	print(f"Fitting KNN imputer with {knn_array.shape[0]} samples")
	_= knn_imputer.fit_transform(knn_array)
	
	return knn_imputer

def get_knn_imputer_predictions(knn_imputer,knn_array):
	print(f"Predicting using KNN imputer on {knn_array.shape[0]} samples")
	predictions_knn= knn_imputer.transform(knn_array)
	
	return predictions_knn

def get_num_cpu_threads():
	"""
	Returns the number of CPU threads available for training in TensorFlow.
	"""
	# get a list of all local devices available to TensorFlow
	local_device_protos = device_lib.list_local_devices()

	# filter out only the CPU devices
	cpu_devices = [x.name for x in local_device_protos if x.device_type == 'CPU']

	# count the number of CPU threads available for training
	num_threads = 0
	for d in cpu_devices:
		num_threads += int(d.split(':')[-1])
	
	return num_threads

def model_to_7ziparchive(archive_file,model_path):
	"""Converted TensorFlow saved model to zip archive"""
	
	archive_file = f"{archive_file}.7z"
	print(f"Converting {model_path} to zip archive at:{archive_file}")
	#with py7zr.SevenZipFile(archive_file, 'w') as archive:
	#	 archive.writeall(model_file)
	
	with py7zr.SevenZipFile(archive_file,'w') as model_file:
		for item in os.listdir(model_path):
			try:
				os.chdir(model_path) #Change directory if folder is not known
				if os.path.isdir(item): #Check if folder and add folder contents
					model_file.write(item)
					for item2 in os.listdir(item):
						#print(f"Adding {item2}..")
						model_file.write(os.path.join(item,item2))
				else:
					#print(f"Adding {item}..")
					#model_file.write(os.path.join(model_folder,item))
					model_file.write(item)
			except:	 
				print(f'File {item} not Found') 

def sevenziparchive_to_model(archive_file,model_folder):
	"""Converted zip archive to TensorFlow saved model"""
	print(f"Extracting pre-trained model {archive_file} to folder:{model_folder}")
	model_path = os.path.join(model_folder,archive_file.split("/")[-1].replace(".7z",""))
	with py7zr.SevenZipFile(archive_file, 'r') as archive:
		archive.extractall(path=model_path)
	
	return model_path

def saved_model_to_tflite(tflitemodel_filename,savedmodel_folder):
	"""Convert TF saved model to tflite model"""
	
	print(f"Converting {savedmodel_folder} to tflite model...")
	#folder_path = os.path.dirname(script_path)
	#file_name_with_extension = os.path.basename(script_path)
	converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_folder) # path to the SavedModel directory
	converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
	#converter.target_spec.supported_types = [tf.float16] #reduce the size of a floating point model by quantizing the weights to float16
	converter._experimental_lower_tensor_list_ops = False
		
	optimize=""
	if optimize=='Speed':
		converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
	elif optimize=='Storage':
		converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
	else:	 
		converter.optimizations = [tf.lite.Optimize.DEFAULT]
	
	tflite_model = converter.convert()
	
	# Save the model.
	print(f"Saving tflite model in:{tflitemodel_filename}")
	with open(f'{tflitemodel_filename}.tflite', 'wb') as f:
		f.write(tflite_model)
