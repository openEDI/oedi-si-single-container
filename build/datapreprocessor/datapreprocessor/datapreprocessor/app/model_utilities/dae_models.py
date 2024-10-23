"""
Created on Wed June 01 00:10:00 2022
@author: splathottam
"""

import glob
import math
import calendar
import os
from typing import List, Set, Dict, Tuple, Optional, Union

import py7zr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers,losses,metrics
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error

print(f"TF version:{tf.__version__}")

class DenoiseFNN(Model):
	def __init__(self):
		super(DenoiseFNN, self).__init__()
		self.encoder = tf.keras.Sequential([
		  layers.Input(shape=(load_block_length)),
		  normalizer,
		  layers.Dense(32, activation="relu"),
		  layers.Dense(16, activation="relu"),
		  layers.Dense(8, activation="relu")])

		self.decoder = tf.keras.Sequential([
		  layers.Dense(16, activation="relu"),
		  layers.Dense(32, activation="relu"),
		  layers.Dense(load_block_length, activation="linear")])

	def call(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded

class Denoise1DCNN(Model):
	def __init__(self,normalizer,load_block_length,n_input_features):
		super(Denoise1DCNN, self).__init__()
		self.encoder = tf.keras.Sequential([
		  layers.Input(shape=(load_block_length, n_input_features)),
		  normalizer, 
		  layers.Conv1D(64, 2, activation='relu'), #'relu'#layers.LeakyReLU()
		  layers.Conv1D(64, 2, activation='relu'),
		  layers.Conv1D(64, 2, activation='relu'),
		  layers.Dense(10, activation='relu')]) #"linear"

		self.decoder = tf.keras.Sequential([
		  layers.Conv1DTranspose(64, kernel_size=2, activation='relu'),
		  layers.Conv1DTranspose(32, kernel_size=2, activation='relu'),
		  layers.Conv1DTranspose(1, kernel_size=2, activation='linear')])

	def call(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		
		return decoded

class Denoise1DCNNDeepHyper(Model):
	def __init__(self,normalizer,load_block_length,n_input_features):
		super(Denoise1DCNN, self).__init__()
		self.encoder = tf.keras.Sequential([
		  layers.Input(shape=(load_block_length, n_input_features)),
		  normalizer, 
		  layers.Conv1D(64, 2, activation='relu'), #'relu'#layers.LeakyReLU()
		  layers.Conv1D(64, 2, activation='relu'),
		  layers.Conv1D(64, 2, activation='relu'),
		  layers.Dense(10, activation='relu')]) #"linear"

		self.decoder = tf.keras.Sequential([
		  layers.Conv1DTranspose(64, kernel_size=2, activation='relu'),
		  layers.Conv1DTranspose(32, kernel_size=2, activation='relu'),
		  layers.Conv1DTranspose(1, kernel_size=2, activation='linear')])

	def call(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded

class Denoise1DCNNNew(Model):
	def __init__(self,normalizer,load_block_length,n_input_features):
		super(Denoise1DCNNNew, self).__init__()
		self.encoder = tf.keras.Sequential([
		  layers.Input(shape=(load_block_length, n_input_features)),
		  normalizer, 
		  layers.Conv1D(64, 2, padding="same", activation='relu'), #'relu'#layers.LeakyReLU()
		  layers.Conv1D(32, 2, padding="same", activation='relu'),
		  layers.Conv1D(32, 2, padding="same", activation='relu'),
		  layers.Conv1D(16, 2, padding="same", activation='relu'),
		  ])

		self.decoder = tf.keras.Sequential([
		  layers.Conv1DTranspose(16, kernel_size=2, padding="same", activation='relu'),
		  layers.Conv1DTranspose(32, kernel_size=2, padding="same", activation='relu'),
		  layers.Conv1DTranspose(32, kernel_size=2, padding="same", activation='relu'),
		  layers.Conv1DTranspose(64, kernel_size=2, padding="same", activation='relu'),
		  layers.Conv1DTranspose(1, kernel_size=2, padding="same", activation='linear')])

	def call(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded

class Denoise1DCNNExtra(Model):
	def __init__(self):
		super(Denoise1DCNNExtra, self).__init__()
		
		self.features = tf.keras.Sequential([
		  layers.Input(shape=(2)),
		  layers.Dense(16, activation="relu",name="cyclical_dense1"),
		  layers.Dense(8, activation="relu",name="cyclical_dense2")])
		
		self.encoder = tf.keras.Sequential([
		  layers.Input(shape=(load_block_length, 2)),
		  #normalizer, 
		  layers.Conv1D(64, 2, activation='relu'),
		  layers.Conv1D(64, 2, activation='relu'),
		  layers.Conv1D(64, 2, activation='relu'),
		  layers.Dense(90, activation="relu")])

		self.decoder = tf.keras.Sequential([
		  layers.Conv1DTranspose(64, kernel_size=2, activation='relu'),
		  layers.Conv1DTranspose(32, kernel_size=2, activation='relu'),
		  layers.Conv1DTranspose(1, kernel_size=2, activation='linear')])

	def call(self, x):
		encoded = self.encoder(x[0])
		cyclical = self.features(x[1])
		concat = layers.Concatenate(name='inputs_concatenate')([encoded,cyclical])
		decoded = self.decoder(concat)
		return decoded

class DenoiseLSTM(Model):
	def __init__(self,normalizer,load_block_length,n_input_features):
		super(DenoiseLSTM, self).__init__()
		self.encoder = tf.keras.Sequential([
		  layers.Input(shape=(load_block_length, n_input_features)),
		  normalizer, 
		  layers.LSTM(100, activation='relu')]) #"linear"

		self.decoder = tf.keras.Sequential([
		  layers.RepeatVector(load_block_length), #RepeatVector layer repeats the incoming inputs a specific number of time
		  layers.LSTM(100, activation='relu', return_sequences=True),
		  layers.TimeDistributed(layers.Dense(1))]) #''#This wrapper allows to apply a layer to every temporal slice of an input.'''

	def call(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded
	
class LSTMAutoEncoder(Model):
	def __init__(self,normalizer,window_size,n_input_features,n_output_features):
		super().__init__()
		self.encoder = tf.keras.Sequential([
		  tf.keras.layers.Input(shape=(window_size, n_input_features)),
		  normalizer, 
		  tf.keras.layers.LSTM(100, activation='relu')]) #"linear"

		self.decoder = tf.keras.Sequential([
		  tf.keras.layers.RepeatVector(window_size), #RepeatVector layer repeats the incoming inputs a specific number of time
		  tf.keras.layers.LSTM(100, activation='relu', return_sequences=True),
		  tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_output_features))]) #This wrapper allows to apply a layer to every temporal slice of an input.'''

	def call(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded

class DenoiseLSTMStateful(Model):
	def __init__(self,normalizer,load_block_length,n_input_features,batch_size):
		super(DenoiseLSTMStateful, self).__init__()
		self.encoder = tf.keras.Sequential([
		  layers.Input(batch_shape= (batch_size, load_block_length, n_input_features)),
		  normalizer, 
		  layers.LSTM(100, activation='relu',stateful=True)]) #"linear"

		self.decoder = tf.keras.Sequential([
		  layers.RepeatVector(load_block_length), #RepeatVector layer repeats the incoming inputs a specific number of time
		  layers.LSTM(100, activation='relu', return_sequences=True,stateful=True),
		  layers.TimeDistributed(layers.Dense(1))]) #''#This wrapper allows to apply a layer to every temporal slice of an input.'''

	def call(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded

class AnomalyLSTM(Model):
	def __init__(self,normalizer,load_block_length,n_input_features):
		super(AnomalyLSTM, self).__init__()
		self.encoder = tf.keras.Sequential([
		  layers.Input(shape=(load_block_length, n_input_features)),
		  normalizer, 
		  layers.LSTM(100, activation='relu'),
		  layers.Dropout(rate=0.2)]) #"linear"

		self.decoder = tf.keras.Sequential([
		  layers.RepeatVector(load_block_length), #RepeatVector layer repeats the incoming inputs a specific number of time
		  layers.LSTM(100, activation='relu', return_sequences=True),
		  layers.Dropout(rate=0.2),
		  layers.TimeDistributed(layers.Dense(1))]) #''#This wrapper allows to apply a layer to every temporal slice of an input.'''

	def call(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded

class BiLSTM(Model):
	def __init__(self,normalizer,load_block_length,n_input_features,n_target_features):
		super(BiLSTM, self).__init__()
		
		self.disaggregator = tf.keras.Sequential([layers.Input(shape=(load_block_length, n_input_features)),
												  normalizer,
												  layers.Bidirectional(layers.LSTM(100, activation='relu',return_sequences=True)),
												  layers.Bidirectional(layers.LSTM(100, activation='relu',return_sequences=True)),
												  layers.TimeDistributed(layers.Dense(1, activation='relu'))]) #"linear"
		
	def call(self, x):
		y = self.disaggregator(x)
		
		return y
	
class BiLSTM_rev1(Model):
	def __init__(self,normalizer,load_block_length,n_input_features,n_target_features):
		super(BiLSTM, self).__init__()
		
		self.disaggregator = tf.keras.Sequential([layers.Input(shape=(load_block_length, n_input_features)),
												  normalizer,
												  layers.Bidirectional(layers.LSTM(100, activation='relu', return_sequences=True)),
												  layers.Bidirectional(layers.LSTM(100, activation='relu', return_sequences=True)),
												  layers.Flatten(),
												  layers.Dense(64, activation="relu"),
												  layers.Dense(16, activation="relu"),
												  
												  layers.Dense(n_target_features, activation="linear")]) #"linear"
		
	def call(self, x):
		y = self.disaggregator(x)
		
		return y
	
class BiLSTMSimple(Model):
	def __init__(self,normalizer,load_block_length,n_input_features,n_target_features):
		super(BiLSTM, self).__init__()
		
		self.disaggregator = tf.keras.Sequential([layers.Input(shape=(load_block_length, n_input_features)),
												  normalizer,
												  layers.Bidirectional(layers.LSTM(64, activation='relu',return_sequences=True)),
												  layers.Dense(64, activation="relu"),
												  layers.Dense(n_target_features, activation="linear")]) #"linear"

		
	def call(self, x):
		y = self.disaggregator(x)
		
		return y

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
	# Normalization and Attention
	x = layers.LayerNormalization(epsilon=1e-6)(inputs)
	x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
	x = layers.Dropout(dropout)(x)
	res = x + inputs

	# Feed Forward Part
	x = layers.LayerNormalization(epsilon=1e-6)(res)
	x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
	x = layers.Dropout(dropout)(x)
	x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
	
	return x + res

def transformer_autoencoder(inputs, head_size, num_heads, ff_dim, bottleneck_dim, dropout=0):
	# Encoder
	x = transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout)
	
	# Bottleneck Layer
	bottleneck = layers.Dense(units=bottleneck_dim, activation="relu")(x)
	x = layers.Dense(units=ff_dim, activation="relu")(bottleneck)
	# Decoder
	x = layers.Reshape((-1, ff_dim))(x)
	x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
	
	# Output
	x = layers.Dense(1)(x)
	
	return x

def get_transformer_ae(normalizer,load_block_length,n_input_features):
	inputs = tf.keras.Input(shape=(load_block_length,n_input_features))

	# Create the autoencoder model
	outputs = transformer_autoencoder(inputs, head_size=256, num_heads=4, ff_dim=4, bottleneck_dim=10, dropout=0.25)
	model = tf.keras.Model(inputs, outputs)
	
	return model

def get_dnn_model(model_type,normalizer,load_block_length,n_input_features,stateful=False):
	"""Return uncompiled model"""
	
	print(f"Selecting model type:{model_type}")
	if model_type == "1dcnn":
		model = Denoise1DCNN(normalizer,load_block_length,n_input_features)

	elif model_type == "lstm":
		print(f"Stateful LSTM:{stateful}")
		if not stateful:
			model = DenoiseLSTM(normalizer,load_block_length,n_input_features)
		else:
			model = DenoiseLSTMStateful(normalizer,load_block_length,n_input_features,batch_size)
	
	elif model_type == "bilstm":
		model = BiLSTM(normalizer,load_block_length,n_input_features,1)
		
	elif model_type == "transformer":
		model = get_transformer_ae(normalizer,load_block_length,n_input_features) #Sequential implementation giving error

	else:
		raise ValueError(f"{model_type} is an invalid model!")
	
	print(f"Compiling model...")
	model.compile(optimizer='adam', loss=losses.MeanSquaredError(),metrics=[metrics.MeanAbsoluteError()]) #losses.MeanAbsoluteError()
    
	return model

def get_normalizer(df,input_features,skip_normalization=[]):
	"""Normalizer"""
	normalizer_means = []
	normalizer_vars = []
	for input_feature in input_features:
		if input_feature not in skip_normalization:
			print(f"Adding mean and vars for {input_feature}")
			normalizer_means.append(df[input_feature].mean())
			normalizer_vars.append(df[input_feature].var())
		else:
			print(f"Adding identities for {input_feature}")
			normalizer_means.append(0)
			normalizer_vars.append(1)
	
	print(f"Means:{normalizer_means}, Vars:{normalizer_vars}")
	normalizer =tf.keras.layers.Normalization(mean=tuple(normalizer_means), variance=tuple(normalizer_vars))	
	
	return normalizer

def get_checkpoint_callback(file_path,monitored_metric):

	#if backend == "kerascore":
	#	model_extension = ".weights.h5" #".keras"
	#else:
	model_extension = ""
	file_path = file_path + model_extension
	print(f"Creating model checkpoint at:{file_path}")
	checkpoint = ModelCheckpoint(filepath= file_path,
                                 monitor=monitored_metric,
                                 verbose=1, 
                                 save_best_only=True,
				 				 save_weights_only=False,
                                 mode="min")
	return checkpoint,model_extension

def plot_training_history(history,plot_filename):
	"""Plot training and validation loss"""
	steps = list(range(1,len(history.history['loss'])+1))
	plt.plot(steps,history.history['loss'],label="training_loss")
	plt.plot(steps,history.history['val_loss'],label="validation_loss")
	
	plt.xlabel("steps")
	plt.ylabel("loss")
	plt.legend()
	plt.savefig(plot_filename)
	plt.show()
