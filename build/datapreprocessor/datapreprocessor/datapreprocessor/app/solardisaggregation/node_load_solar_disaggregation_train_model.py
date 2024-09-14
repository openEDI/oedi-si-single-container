"""'
Created on Friday March 17 11:00:00 2023
@author: Siby Plathottam
"""
import os
import sys
import random
import argparse
import calendar

import pandas as pd
import tensorflow as tf

from numpy.random import default_rng

baseDir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  #Add path of home directory e.g.'/home/splathottam/GitHub/oedi'
workDir=os.path.join(baseDir,"oedianl")

print(f"Adding home directory:{baseDir} to path")
sys.path.insert(0,baseDir) #Add module path to prevent import errors

from datapreprocessor.app.nodeload.smartmeter_data_preprocessing import valid_load_types_v2,get_time_series_dataframe,valid_load_types
from datapreprocessor.app.nodeload.timeseries_data_utilities import get_config_dict
from datapreprocessor.app.nodeload.nodeload_preprocessing import encode_cyclical_features
from datapreprocessor.app.nodeload.datapipeline_utilities import get_input_target_dataset,check_moving_window,get_train_test_eval_nodes,df_to_input_target_dataset
from datapreprocessor.app.nodeload.nodeload_utilities import check_and_create_folder,get_upsampled_df
from datapreprocessor.app.model_utilities.model_utilities import get_autoencoder_model,get_compiled_model,get_checkpoint_callback,get_normalizer,evaluate_predict
from datapreprocessor.app.model_utilities.model_training_utilities import train_model,get_best_model
from datapreprocessor.app.model_utilities.model_save_load_utilities import model_to_archive,load_keras_model
from datapreprocessor.app.solardisaggregation.solardisaggregation_preprocessing import convert_solardata_to_timeseries,generate_solar_node_profiles,get_df_train_solar_disaggregation

rng = default_rng()

## Specify locations of plots and models
folder_plots = os.path.join(baseDir,"datapreprocessor","app","solardisaggregation","plots")
folder_model_inference = os.path.join(baseDir,"datapreprocessor","app","solardisaggregation","model")
folder_model_archive = os.path.join(baseDir,"datapreprocessor","app","solardisaggregation","model_archives")
folder_model_checkpoints = os.path.join(baseDir,"datapreprocessor","app","solardisaggregation","model_checkpoints")
check_and_create_folder(folder_plots)
check_and_create_folder(folder_model_inference)
check_and_create_folder(folder_model_archive)
check_and_create_folder(folder_model_checkpoints)

parser=argparse.ArgumentParser()
parser.add_argument('-c','--config',help='config to be passed to the solar disaggregation training script',default = "solar_disaggregation_config.json", required=False)
args=parser.parse_args()
config_file = args.config

config_file = os.path.join(workDir,"app","solardisaggregation",config_file)
config_dict= get_config_dict(config_file)

## Select timeseries file for use as base file
selected_timeseries_file = os.path.join(workDir,config_dict["nodeload_data_details"]["selected_timeseries_file"]) #Specify file containing time series solar home data

## Specify details of anonymized node load profiles
upsample_original_time_series = config_dict["nodeload_data_details"]["upsample_original_time_series"] # Should the original time series be upsampled
upsample_time_period = config_dict["nodeload_data_details"]["upsample_time_period"] #"Time period of upsampling
selected_months = config_dict["nodeload_data_details"]["selected_months"] #2 # The month for which we are developing the model
distribution_system = config_dict["nodeload_data_details"]["distribution_system"] ##The distribution system we are generating the profiles
distribution_system_file = config_dict["nodeload_data_details"]["distribution_system_file"] ##The opendss file
measurement_column = config_dict["nodeload_data_details"]["measurement_column"]
opendss_casefile = os.path.join(baseDir,"datapreprocessor","data",distribution_system_file)
load_scaling_mode = config_dict["nodeload_data_details"]["load_scaling_mode"] #"simple" #multi
n_customers = config_dict["nodeload_data_details"]["n_customers"] #300
max_solar_penetration = config_dict["nodeload_data_details"]["max_solar_penetration"] #0.3 #The maximum solar penetration at a node
month_names = '-'.join([calendar.month_abbr[num] for num in selected_months])

## Training data specifications
n_days = config_dict["train_data_details"]["n_days"] #4 #The number of full day profiles that will be generated for training
n_nodes = config_dict["train_data_details"]["n_nodes"] #4 #The number of nodes that will be used for training
cyclical_features = config_dict["train_data_details"]["cyclical_features"] #["hour_of_day",'day_of_week','weekend'] #Cyclical features to be added to the training data

## Input/target features for data imputation model
encoded_cyclical_features= ['cos_hour','sin_hour','cos_day_of_week','sin_day_of_week']#,'weekend']

auxiliary_features = []
input_features = [f"{measurement_column}_corrupted"] + auxiliary_features  + ["corruption_encoding"]  + encoded_cyclical_features
target_feature =  f"{measurement_column}"
n_input_features = len(input_features)
n_target_features =  1
print(f"Using following {n_input_features} features as input to data imputation model:{input_features}")

input_features = ["net_load"] + encoded_cyclical_features #gross_load
target_feature =  "solar_power"
target_features = ["solar_power"]
n_input_features = len(input_features)
print(f"Using following {n_input_features} features as input to data imputation model:{input_features}")

## Generated base solar profiles from solar data
df_solar_timeseries = pd.read_csv(selected_timeseries_file, parse_dates=['datetime'])

## Generate node solar profiles for the distribution system model we are intrested in
df_solar_node,solar_node_dict = generate_solar_node_profiles(df_solar_timeseries,opendss_casefile,selected_months,n_solar_nodes=n_nodes,max_solar_penetration = max_solar_penetration,upsample_time_series=upsample_original_time_series,upsample_time_period=upsample_time_period)





## Specify train and test nodes
selected_train_nodes,selected_test_nodes,selected_eval_nodes = get_train_test_eval_nodes(solar_node_dict,train_fraction=0.8,test_fraction=0.2)

## Generate training and testing data
df_train = get_df_train_solar_disaggregation(df_solar_node,cyclical_features,selected_train_nodes)
df_test = get_df_train_solar_disaggregation(df_solar_node,cyclical_features,selected_test_nodes)
n_train_samples = len(df_train)

## Convert dataframe into a dataset object that can be used by model training
train_input_target = df_to_input_target_dataset(df_train,load_block_length,input_features,target_feature,batch_size,use_prefetch,df_type = "train")
test_input_target = df_to_input_target_dataset(df_test,load_block_length,input_features,target_feature,batch_size,use_prefetch,df_type = "test")

## Create object to normalize data
normalizer = get_normalizer(df_train,input_features,skip_normalization=encoded_cyclical_features) #Obtain a normalizer using training data
print(f"Raw data:{test_input_target.take(1).as_numpy_iterator().next()[0][0:2]}")
print(f"NOrmalized data:{normalizer(test_input_target.take(1).as_numpy_iterator().next()[0][0:2])}")

## Create solar disaggregation model
predictor = get_dnn_model(model_type,normalizer,load_block_length,n_input_features)
#predictor.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(),metrics=[tf.keras.metrics.MeanAbsoluteError()]) #losses.MeanAbsoluteError()

monitored_metric = "val_loss"#"val_mean_absolute_error" #"val_loss"
checkpoint_file = 'disag_multi_model.epoch{epoch:02d}-loss{val_loss:.5f}' #'-mae{val_mean_absolute_error:.5f}'
checkpoint =  tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(folder_name_saved_models,f'm-{month_names}_w-{load_block_length}_f-{n_input_features}_msp-{max_solar_penetration}_n-{n_train_samples}_{model_type}_'+checkpoint_file),
                             monitor=monitored_metric,
                             verbose=1, 
                             save_best_only=True,
                             mode="min")

callbacks = [checkpoint]

## Train model
tic = time.perf_counter()
history = predictor.fit(train_input_target,epochs=n_epochs,shuffle=True,validation_data=test_input_target,callbacks=callbacks)
toc = time.perf_counter()

plot_training_history(history,os.path.join(folder_name_plots,f"loss_history_{distribution_system}_m-{month_names}_w-{load_block_length}_f-{n_input_features}_n-{n_train_samples}_{model_type}.png"))
predictor.summary(expand_nested=True)
print(f"Training took:{toc-tic} s")

## Convert best saved model into a 7z file which can be used by the inference script
best_val_loss = min(history.history['val_loss'])
best_epoch = history.history['val_loss'].index(best_val_loss) +1

prediction_model_folder = os.path.join(folder_name_saved_models,f'm-{month_names}_w-{load_block_length}_f-{n_input_features}_msp-{max_solar_penetration}_n-{n_train_samples}_{model_type}_disag_multi_model.epoch{best_epoch:02d}-loss{best_val_loss:.5f}')

model_archive = os.path.join(workDir,"app","solardisaggregation","model",f'disagg_model-{model_type}_dss-{distribution_system}_m-{month_names}_w-{load_block_length}_f-{n_input_features}_{model_identifier}.7z')
assert model_type in prediction_model_folder, f"model type:{model_type} not found in {prediction_model_folder}"
model_to_7ziparchive(model_archive,prediction_model_folder)
