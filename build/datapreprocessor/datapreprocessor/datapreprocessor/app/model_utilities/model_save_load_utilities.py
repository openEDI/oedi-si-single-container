"""
Created on Febrary 15 11:00:00 2023
@author: Siby Plathottam
"""

import os
import shutil

import keras
import py7zr
import tensorflow as tf

def check_keras_model_path(model_path:str):
	assert '.keras' in model_path, f"{model_path} needs to to be a .keras file"

def save_keras_model(model,model_path:str):
	check_keras_model_path(model_path)
	print(f"Saving Keras model at:{model_path}")
	keras.saving.save_model(model, model_path, overwrite=True)

def save_tfkeras_model(model,model_path:str):
	print(f"Saving Keras model as TensorFlow SavedModel format:{model_path}")
	#tf.keras.saving.save_model(model, model_path, overwrite=True, save_format='tf')
	tf.saved_model.save(model, model_path)

def load_keras_model(model_path:str,custom_objects:dict=None):
	check_keras_model_path(model_path)
	print(f"Loading Keras model:{model_path}")
	if custom_objects is not None:
		print(f"Using following custom objects:{custom_objects}")
	model = keras.models.load_model(filepath=model_path,custom_objects = custom_objects)
	print("Successfully loaded Keras model!")
	return model

def load_keras_model_weights(model,weights_path:str):

	assert 'weights.h5' in weights_path, f"{weights_path} needs to to be a .weights.h5 file"
	print(f"Loading weights from:{weights_path}")
	model.load_weights(weights_path)
	
	return model

def load_tfkeras_model(model_path:str):
	print(f"Loading Keras model saved in TensorFlow SavedModel format:{model_path}")
	model = tf.keras.models.load_model(model_path)

	return model

def load_tf_savedmodel(model_path:str):
	print(f"Loading TF SavedModel:{model_path}")
	model = tf.saved_model.load(model_path)
	
	return model

def model_to_archive(model_path:str,model_archivepath:str):
	"""Converted saved Keras/TensorFlow model to zip archive"""

	if ".keras" in model_path:
		model_archivepath  = model_archivepath + ".keras"
		print(f"Saving model as Keras archive in:{model_archivepath}")
		shutil.copy(model_path, model_archivepath)	
	else:
		model_archivepath = model_to_7ziparchive(model_path,model_archivepath)
	
	return model_archivepath

def model_to_7ziparchive(model_path:str,model_archivepath:str):
	"""Converted saved Keras/TensorFlow model to zip archive"""
	
	model_archivepath = f"{model_archivepath}.7z"
	print(f"Converting {model_path} to zip archive at:{model_archivepath}")
	#with py7zr.SevenZipFile(archive_file, 'w') as archive:
	#	 archive.writeall(model_savepath)
	
	with py7zr.SevenZipFile(model_archivepath,'w') as model_archive:
		if os.path.isfile(model_path): #check if model defininion is a file
			#model_archive.write(model_path)
			model_archive.write(model_path,arcname=os.path.basename(model_archivepath)) #Only create archive from the model file and not the full path
		elif os.path.isdir(model_path): #check if model defininion is a folder
			for item in os.listdir(model_path):
				try:
					os.chdir(model_path) #Change directory if folder is not known
					if os.path.isdir(item): #Check if folder and add folder contents
						model_archive.write(item)
						for item2 in os.listdir(item):						
							model_archive.write(os.path.join(item,item2))
					else:					
						model_archive.write(item)
				except:	 
					print(f'File {item} not Found')
		else:
			raise ValueError(f"{model_path} is an invalid model save path!")

	return model_archivepath

def modelarchive_to_modelpath(model_archivepath:str,model_folder:str):
	"""Convert model archive path to model path"""
	
	print(f"Extracting pre-trained model {model_archivepath} to folder:{model_folder}")
	
	if ".7z" in model_archivepath: #Check if it is a 7z archive
		model_path = os.path.join(model_folder,model_archivepath.split("/")[-1].replace(".7z",""))
		with py7zr.SevenZipFile(model_archivepath, 'r') as archive:
			archive.extractall(path=model_path)
	elif ".keras" in model_archivepath: #Check if it is a keras archive
		_, model_path = os.path.split(model_archivepath)		
		model_path = os.path.join(model_folder,model_path) #extract file name and add it to model folder		
		shutil.copy(model_archivepath, model_folder)
	else:
		raise ValueError(f"{model_archivepath} is not a valid model archive!")	
	
	return model_path
