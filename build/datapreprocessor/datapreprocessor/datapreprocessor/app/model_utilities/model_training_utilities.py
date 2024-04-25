"""
Created on January 3 3:00:00 2023
@author: Siby Plathottam
"""

import os
import time
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import keras

from datapreprocessor.datapreprocessor.app.model_utilities.model_save_load_utilities import load_keras_model

def train_model(model,train_dataset,validation_dataset,n_epochs:int,callbacks:List):
	"""Train the model and return model+ train history"""

	print(f"Training using following Keras backend:{keras.backend.backend()}")
	tic = time.perf_counter()
	history = model.fit(train_dataset,validation_data=validation_dataset,epochs=n_epochs,callbacks=callbacks) #Train model
	toc = time.perf_counter()

	print(f"Training time for {n_epochs} epochs:{toc-tic:.3f} s")
	print(f"Training time per epoch:{(toc-tic)/n_epochs:.3f} s")

	model.summary(expand_nested=True,show_trainable=True)
	plot_training_history(history,plot_filename="plots/model_training_progress.png")

	return model,history

def plot_training_history(history,plot_filename:str):
	"""Plot training and validation loss"""

	print(f"Saving training history plot in {plot_filename}...")
	steps = list(range(1,len(history.history['loss'])+1))
	plt.plot(steps,history.history['loss'],label="training_loss")
	plt.plot(steps,history.history['val_loss'],label="validation_loss")
	
	plt.xlabel("steps")
	plt.ylabel("loss")
	plt.legend()
	plt.savefig(plot_filename)
	plt.show()

def get_best_model(history,monitored_metric:str):
	best_monitored_metric = min(history.history[monitored_metric])
	best_epoch = history.history[monitored_metric].index(best_monitored_metric) +1
	print(f"Best model found at epoch:{best_epoch} with {monitored_metric}:{best_monitored_metric:4f}")
	
	return best_monitored_metric,best_epoch

def check_saved_model(model,model_path:str,input_data):	
	loaded_model = load_keras_model(model_path)
	assert np.allclose(model.predict(input_data), loaded_model.predict(input_data)), "Model is different from loaded model!"

