"""
Created on December 29 10:00:00 2023
@author: Siby Plathottam
"""

import keras

from datapreprocessor.app.model_utilities.models import AutoEncoder1DCNN,LSTMAutoEncoder

def get_compiled_model(model):

	print(f"Compiling model using Keras backend:{keras.backend.backend()}...")
	model.compile(optimizer=keras.optimizers.AdamW(5e-5), loss=keras.losses.MeanSquaredError(),metrics=[keras.metrics.MeanAbsoluteError()], #keras.metrics.RootMeanSquaredError()
			     jit_compile=True,) #losses.MeanAbsoluteError()
	
	model.summary(expand_nested=True,show_trainable=True)
	
	return model

def get_model(inputs,outputs):

	model = keras.Model(inputs=inputs, outputs=outputs)
	model = get_compiled_model(model)
	
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
	normalizer = keras.layers.Normalization(mean=tuple(normalizer_means), variance=tuple(normalizer_vars))	
	
	return normalizer

def get_autoencoder_model(model_type,window_size,n_input_features,n_output_features,normalizer=None,**kwargs):
	"""Return uncompiled model"""
	
	print(f"Selecting autoencoder model type:{model_type}")
	if model_type.lower() == "1dcnn":
		model = AutoEncoder1DCNN(window_size,n_input_features,n_output_features,normalizer,**kwargs)

	elif model_type.lower() == "lstm":
		model = LSTMAutoEncoder(window_size,n_input_features,n_output_features,normalizer,**kwargs)
	
	#elif model_type.lower() == "transformer":
	#	model = TransformerAutoEncoder(window_size,n_input_features,**kwargs)
	
	else:
		raise ValueError(f"{model_type} is an invalid model!")	
	print(f"Returning model of type:{type(model)}")
	
	return model

def get_checkpoint_callback(model_checkpoint_path,monitored_metric:str,save_weights_only:bool=False):

	if save_weights_only:
		model_extension = ".weights.h5"
	else:
		model_extension = ".keras"
	
	print(f"Creating model checkpoint at:{model_checkpoint_path + model_extension}")
	
	checkpoint = keras.callbacks.ModelCheckpoint(filepath = model_checkpoint_path + model_extension,
                                 monitor=monitored_metric,
                                 verbose=1, 
                                 save_best_only=True,
				 				 save_weights_only=save_weights_only,
                                 mode="min")
	
	return checkpoint

def evaluate_predict(model,input_target=None,data_dict=None):
	"""Evaluate and predict using dataset or dictionary"""
		
	if input_target is not None:
		model_performance_metrics = model.evaluate(input_target)
		predictions = model.predict(input_target)
	elif data_dict is not None:
		model_performance_metrics = model.evaluate(data_dict["features"],data_dict["target"])
		predictions = model.predict(data_dict["features"])
	else:
		raise ValueError("Either tf dataset or dict should be provided")
		
	print(f"Model performance metrics:{model_performance_metrics}")	   
	print(f"Predictions shape:{predictions.shape}")
	
	return predictions
