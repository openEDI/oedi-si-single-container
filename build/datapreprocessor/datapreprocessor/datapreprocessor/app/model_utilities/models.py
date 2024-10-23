"""
Created on December 29 10:00:00 2023
@author: Siby Plathottam
"""

import keras
import tensorflow as tf

tf.experimental.numpy.experimental_enable_numpy_behavior()
keras.mixed_precision.set_global_policy("float64")	# Use mixed precision to speed up all training #mixed_float16

@keras.saving.register_keras_serializable(package="1DCNNAutoEncoder") #This is required for reloading Keras model
class AutoEncoder1DCNN(keras.Model): # Create model by subclassing the Model class
	def __init__(self, window_size: int, n_input_features: int, n_output_features: int, normalizer=None, *args, **kwargs):
		super().__init__(*args, **kwargs)  # This is sufficient since we are inheriting from single class
		
		self.window_size = window_size
		self.n_input_features = n_input_features
		self.n_output_features = n_output_features
		self.normalizer = normalizer

		encoder_layers =  [keras.Input(shape=(window_size, n_input_features))]
		if normalizer is not None:
			encoder_layers.append(normalizer)
		encoder_layers.extend([keras.layers.Conv1D(64, 2, activation='relu'),  # 'relu'#layers.LeakyReLU()
							   keras.layers.Conv1D(64, 2, activation='relu'),
							   keras.layers.Conv1D(64, 2, activation='relu'),
							   keras.layers.Dense(10, activation='relu')]
							 )
		self.encoder = keras.Sequential(encoder_layers, name="encoder") # "linear"
		self.decoder = keras.Sequential([keras.layers.Conv1DTranspose(64, kernel_size=2, activation='relu'),
										 keras.layers.Conv1DTranspose(32, kernel_size=2, activation='relu'),
										 keras.layers.Conv1DTranspose(n_output_features, kernel_size=2, activation='linear')], name="decoder")

	def call(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)

		return decoded
	
	def get_config(self): #This is required for saved models to load properly with normalizer
		base_config = super().get_config()
		config = {"window_size": self.window_size, "n_input_features": self.n_input_features,
				  "n_output_features": self.n_output_features,"normalizer": keras.saving.serialize_keras_object(self.normalizer)}

		return {**base_config, **config}
	
	@classmethod
	def from_config(cls, config): #This is required for saved models to load properly with normalizer
		config["normalizer"] = keras.layers.deserialize(config["normalizer"])
		return cls(**config)

@keras.saving.register_keras_serializable(package="LSTMAutoEncoder")
class LSTMAutoEncoder(keras.Model):
	def __init__(self, window_size: int, n_input_features: int, n_output_features: int, normalizer=None, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.window_size = window_size
		self.n_input_features = n_input_features
		self.n_output_features = n_output_features
		self.normalizer = normalizer

		if normalizer is not None:
			self.encoder = keras.Sequential([
				keras.Input(shape=(window_size, n_input_features)),
				normalizer,
				keras.layers.LSTM(100, activation='relu')])	 # "linear"
		else:
			self.encoder = keras.Sequential([
				keras.Input(shape=(window_size, n_input_features)),
				keras.layers.LSTM(100, activation='relu')])	 # "linear"

		self.decoder = keras.Sequential([
			keras.layers.RepeatVector(window_size), # RepeatVector layer repeats the incoming inputs a specific number of time
			keras.layers.LSTM(100, activation='relu', return_sequences=True),
			keras.layers.TimeDistributed(keras.layers.Dense(n_output_features))])  #TimeDistributed wrapper allows to apply a layer to every temporal slice of an input.'''

	def call(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)

		return decoded

	def get_config(self): #This is required for saved models to load properly with normalizer
		base_config = super().get_config()
		config = {"window_size": self.window_size, "n_input_features": self.n_input_features,
				  "n_output_features": self.n_output_features,"normalizer": keras.saving.serialize_keras_object(self.normalizer)}

		return {**base_config, **config}
	
	@classmethod
	def from_config(cls, config): #This is required for saved models to load properly with normalizer
		config["normalizer"] = keras.layers.deserialize(config["normalizer"])
		return cls(**config)

class TransformerAutoEncoder(keras.Model):
	def __init__(self,normalizer,window_size:int,n_input_features:int,n_transformer_layers:int=1,heads_per_transformer:int=2,*args,**kwargs):
		print(f"Building transformer autoencoder with {n_transformer_layers} layers with {heads_per_transformer} heads per transformer...")
		super().__init__(*args,**kwargs) #This is sufficient since we are inheriting from single class

		encoder_layers = [keras.Input(shape=(window_size, n_input_features)),normalizer]
		decoder_layers = []

		for _ in range(n_transformer_layers):
			encoder_layers.append(get_transformer_encoder(num_heads=heads_per_transformer))

		for _ in range(n_transformer_layers):
			decoder_layers.append(get_transformer_decoder(num_heads=heads_per_transformer))

		self.encoder = keras.Sequential(encoder_layers,name = "encoder")
		self.decoder = keras.Sequential(decoder_layers,name = "decoder")

	def call(self, x):
		encoder_output = self.encoder(x)
		decoder_output = self.decoder(encoder_output)

		return decoder_output

def get_transformer_encoder(num_heads:int=2):

	encoder = keras_nlp.layers.TransformerEncoder(intermediate_dim=64, num_heads=num_heads,dropout=0.1,) # Create a single transformer encoder layer.

	#outputs = keras_nlp.layers.TransformerEncoder(num_heads=num_heads,intermediate_dim=128,dropout=0.1,)(outputs)
	#outputs = keras.layers.Dense(2)(outputs[:, 0, :])
	#model = keras.Model(inputs=token_id_input,outputs=outputs,)

	return encoder

def get_transformer_decoder(num_heads:int=2):
	decoder = keras_nlp.layers.TransformerDecoder(intermediate_dim=64, num_heads=num_heads) # Create a single transformer decoder layer.

	return decoder

@keras.saving.register_keras_serializable(package="BiLSTMRegressor")
class BiLSTMRegressor(keras.Model):
	def __init__(self, window_size: int, n_input_features: int, n_target_features: int, normalizer=None, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.window_size = window_size
		self.n_input_features = n_input_features
		self.n_target_features = n_target_features
		self.normalizer = normalizer

		bilstm_layers =	 [keras.Input(shape=(window_size, n_input_features))]
		if normalizer is not None:
			bilstm_layers.append(normalizer)
		bilstm_layers.extend([keras.layers.Bidirectional(keras.layers.LSTM(100, activation='relu',return_sequences=True)),
							  keras.layers.Bidirectional(keras.layers.LSTM(100, activation='relu',return_sequences=True)),
							  keras.layers.TimeDistributed(keras.layers.Dense(n_target_features, activation='relu'))]
							)
		self.bilstm = keras.Sequential(bilstm_layers, name="bilstmregressor")
			
	def call(self, x):
		y = self.bilstm(x)
		
		return y
	
	def get_config(self): #This is required for saved models to load properly with normalizer
		base_config = super().get_config()
		config = {"window_size": self.window_size, "n_input_features": self.n_input_features,
				  "n_target_features": self.n_target_features,"normalizer": keras.saving.serialize_keras_object(self.normalizer)}

		return {**base_config, **config}
	
	@classmethod
	def from_config(cls, config): #This is required for saved models to load properly with normalizer
		config["normalizer"] = keras.layers.deserialize(config["normalizer"])		 
		return cls(**config)

@keras.saving.register_keras_serializable(package="LSTMRegressor")
class LSTMRegressor(keras.Model):
	def __init__(self, window_size: int, n_input_features: int, n_output_features: int, normalizer=None, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.window_size = window_size
		self.n_input_features = n_input_features
		self.n_output_features = n_output_features
		self.normalizer = normalizer

		if normalizer is not None:
			self.encoder = keras.Sequential([
				keras.Input(shape=(window_size, n_input_features)),
				normalizer,
				keras.layers.LSTM(100, activation='relu',return_sequences=True),
				keras.layers.TimeDistributed(keras.layers.Dense(n_output_features, activation='relu'))])	 # "linear"
		else:
			self.encoder = keras.Sequential([
				keras.Input(shape=(window_size, n_input_features)),
				keras.layers.LSTM(100, activation='relu',return_sequences=True),
				keras.layers.TimeDistributed(keras.layers.Dense(n_output_features, activation='relu'))])	 # "linear" #TimeDistributed wrapper allows to apply a layer to every temporal slice of an input.'''

	def call(self, x):
		y = self.encoder(x)
		
		return y

	def get_config(self): #This is required for saved models to load properly with normalizer
		base_config = super().get_config()
		config = {"window_size": self.window_size, "n_input_features": self.n_input_features,
				  "n_output_features": self.n_output_features,"normalizer": keras.saving.serialize_keras_object(self.normalizer)}

		return {**base_config, **config}
	
	@classmethod
	def from_config(cls, config): #This is required for saved models to load properly with normalizer
		config["normalizer"] = keras.layers.deserialize(config["normalizer"])
		return cls(**config)