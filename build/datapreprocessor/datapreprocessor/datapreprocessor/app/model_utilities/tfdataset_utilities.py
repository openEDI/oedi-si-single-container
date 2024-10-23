"""
Created on December 20 11:00:00 2023
@author: Siby Plathottam
"""

import time
import collections
from typing import List,Union

import keras
import tensorflow as tf
import numpy as np

from tqdm import tqdm

def get_tfdataset_element(dataset):

	return dataset.take(1).as_numpy_iterator().next()						  

def show_tfdataset_element(dataset,dataset_name):
	"""Show single element in a TF dataset as numpy array"""
	
	print(f"{dataset_name}:")
	print(f"Element spec:{dataset.element_spec}")
	dataset_element = get_tfdataset_element(dataset)
	
	if isinstance(dataset_element,tuple): #For zipped dataset
		for i,dataset_element_ in enumerate(dataset_element):
			print(f"Tuple {i} - Dataset element shape:{dataset_element_.shape}")
			print(f"Tuple {i} - Dataset element:{dataset_element_}")
	elif isinstance(dataset.element_spec, collections.OrderedDict): #For csv dataset
		for feature,data in dataset_element.items():
			print(f"{feature}:{data}")
	else:		
		print(f"Dataset element:{dataset_element}")

def compare_tfdataset_elements(dataset_dict):
	print(f"Comparing following datasets:{list(dataset_dict.keys())}")
	for dataset_name,dataset in dataset_dict.items():		
		show_tfdataset_element(dataset,dataset_name)	

def show_tfdataset_cardinatlity(dataset,dataset_name):
	"""Show number of elements"""

	print(f"Cardinality for {dataset_name}:{tf.data.experimental.cardinality(dataset).numpy()}")

def benchmark_tfdataset(dataset, num_epochs:int=2,dataset_name:str=""):
	"""Benchmark a TF dataset"""
	
	print(f"Running performance benchmark on tfdataset:{dataset_name} for {num_epochs} epochs...")
	tic = time.perf_counter()
	for _ in range(num_epochs):
		for _ in tqdm(dataset):
			# Performing a training step
			#time.sleep(0.001)
			pass
	toc = time.perf_counter()
	print(f"Execution time:{(toc-tic)/num_epochs:.3f} s/epoch")

def tfdataset_to_numpyarray(dataset,n_elements:int=None,concatenate=False,dataset_name:str=""):
	"""Convert tfdataset to numpy array"""
	
	if n_elements:
		print(f"Converting {n_elements} elements of dataset:{dataset_name} to Numpy array....")
		numpyarray_list = list(dataset.take(n_elements).as_numpy_iterator())
	else:
		print(f"Converting all elements of dataset:{dataset_name} to Numpy array....")
		numpyarray_list = list(dataset.as_numpy_iterator()) #Convert all elements
	if concatenate:
		numpyarray = np.concatenate(numpyarray_list, axis=0)
	else:
		numpyarray = np.array(numpyarray_list)
	print(f"Numpy array shape:{numpyarray.shape}")
	
	return numpyarray

def get_normalizer_from_tfdataset(dataset,features:List[str],n_elements:Union[int,None]=None,skip_normalization:List[str]=[]):
	"""Normalizer"""

	normalizer_means = []
	normalizer_vars = []
	dataset_array = tfdataset_to_numpyarray(dataset,n_elements,dataset_name="dataset_for_adapting_normalizer")
	assert dataset_array.shape[-1] == len(features), "Number of features mismatched."
	print(f"Calcuating means and variances from {dataset_array.shape[0]} samples...")

	for i,input_feature in enumerate(features):
		if input_feature not in skip_normalization:
			print(f"Adding mean and vars for {input_feature}")			
			normalizer_means.append(dataset_array[:,i].mean(axis=0))
			normalizer_vars.append(dataset_array[:,i].var(axis=0))
		else:
			print(f"Adding identities for {input_feature}")
			normalizer_means.append(0.0)
			normalizer_vars.append(1.0)
	
	print(f"Means:{normalizer_means}, Vars:{normalizer_vars}")
	normalizer = keras.layers.Normalization(mean=tuple(normalizer_means), variance=tuple(normalizer_vars))	
	
	check_normalizer(normalizer,dataset)

	return normalizer

def check_normalizer(normalizer,dataset):
	print("Checking normalizer...")
	data = get_tfdataset_element(dataset)
	print(f"Data shape:{data.shape}")
	print(f"Raw data:{data}")
	print(f"Normalized data:{normalizer(data)}")
