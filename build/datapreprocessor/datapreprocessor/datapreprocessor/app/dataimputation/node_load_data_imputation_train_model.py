"""'
Created on Thursday Feb 20 15:00:00 2023
@author: Siby Plathottam
"""
import os
import sys
import glob
import random
import argparse
import calendar
import time

import tensorflow as tf

from numpy.random import default_rng

baseDir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  #Add path of home directory e.g.'/home/splathottam/GitHub/oedi'
workDir=os.path.join(baseDir,"oedianl")

print(f"Adding home directory:{baseDir} to path")
sys.path.insert(0,baseDir) #Add module path to prevent import errors

from oedianl.app.nodeload.timeseries_data_utilities import get_time_series_dataframe,combine_time_series_loadtype_months,get_statistics,get_config_dict
from oedianl.app.nodeload.nodeload_utilities import create_average_timeseries_profiles,generate_load_node_profiles,check_and_create_folder
from oedianl.app.nodeload.nodeload_preprocessing import encode_cyclical_features
from oedianl.app.nodeload.datapipeline_utilities import get_input_target_dataset,check_moving_window,get_train_test_eval_nodes,df_to_input_target_dataset
from oedianl.app.dataimputation.data_imputation_preprocessing import get_df_node_load_selected_nodes,get_replace_nans
from oedianl.app.dataimputation.data_imputation_postprocessing import compare_performance_moving_window
from oedianl.app.dataimputation.dae_models import get_dnn_model,get_normalizer,plot_training_history
from oedianl.app.dataimputation.model_utilities import get_num_cpu_threads,load_evaluate_predict,model_to_7ziparchive,saved_model_to_tflite
from oedianl.app.dopf.model.distmodel import DistModel
rng = default_rng()

folder_name_timeseries = os.path.join(workDir,"data")#,"smartmeter")
folder_name_plots = os.path.join(workDir,"app","dataimputation","plots")

parser=argparse.ArgumentParser()
parser.add_argument('-c','--config',help='config to be passed to the data imputation training script',default = "data_imputation_config.json", required=False)
args=parser.parse_args()
config_file = args.config

config_file = os.path.join(workDir,"app","dataimputation",config_file)
config_dict= get_config_dict(config_file)

selected_timeseries_files  = config_dict["nodeload_data_details"]["selected_timeseries_files"] #["2016_11_60062_time_series.csv"] #Specify file containing zip code level time series data from individual smart meters #"2016_02_60621_time_series.csv"
selected_month = config_dict["nodeload_data_details"]["selected_month"] #2 # The month for which we are developing the model
distribution_system = config_dict["nodeload_data_details"]["distribution_system"] #"123Bus"
distribution_system_file = config_dict["nodeload_data_details"]["distribution_system_file"] #"123Bus"
upsample_original_time_series = config_dict["nodeload_data_details"]["upsample_original_time_series"] # True
upsample_time_period = config_dict["nodeload_data_details"]["upsample_time_period"] #"15Min"
load_scaling_type = config_dict["nodeload_data_details"]["load_scaling_type"] #"simple" #multi

corrupted_fraction = config_dict["train_data_details"]["corrupted_fraction"]  #0.05 #The minimum fraction of values that will be missing
consequtive_corruption_probabilities = config_dict["train_data_details"]["consequtive_corruption_probabilities"]  #{"two":{"conditional_probability":0.2},"three":{"conditional_probability":0.1}}
n_days = config_dict["train_data_details"]["n_days"] #4 #The number of full day profiles that will be generated for training
n_nodes = config_dict["train_data_details"]["n_nodes"] #4 #The number of nodes that will be used for training
load_block_length = config_dict["train_data_details"]["load_block_length"] #4 #The lenghth of the time window

model_type = config_dict["model_arch_details"]["model_type"] #"lstm" #"1dcnn"#"lstm" #Currently enther lstm or 1dcnn
stateful = False #True #False

use_prefetch= True
batch_size =  config_dict["model_training_details"]["batch_size"] #32
n_epochs =  config_dict["model_training_details"]["n_epochs"] #5
model_identifier = config_dict["model_training_details"]["model_identifier"]  #"v0"

selected_timeseries_files = [os.path.join(folder_name_timeseries,timeseries_file) for timeseries_file in selected_timeseries_files]
opendss_casefile = os.path.join(workDir,"data","opendss",distribution_system_file)
folder_name_saved_models = os.path.join(workDir,"app","dataimputation","saved_models",distribution_system,f'month_{calendar.month_abbr[selected_month]}')

check_and_create_folder(folder_name_plots)
check_and_create_folder(folder_name_saved_models)

## Generated averaged load profiles for all load type within the selected time series file
df_averaged_load,df_averaged_day_load = create_average_timeseries_profiles(timeseries_files=selected_timeseries_files,month=selected_month,convert_to_kW=True,upsample=upsample_original_time_series,upsample_time_period=upsample_time_period)

## Generate node load profiles for the distribution system model we are intrested in
df_node_load,load_node_dict = generate_load_node_profiles(df_averaged_day_load,case_file=opendss_casefile,n_nodes=n_nodes,n_days=n_days,start_year = 2016,start_month=selected_month,start_day=1,scaling_type=load_scaling_type)

## Select features for model training
replacement_methods=["ffill"]#,"bfill","mean","median","LI"]

cyclical_features = ["hour_of_day",'day_of_week','weekend']
encoded_cyclical_features= ['cos_hour','sin_hour']#,'cos_day_of_week','sin_day_of_week','weekend']
auxiliary_features = ['load_value_corrupted_ffill']#,'load_value_corrupted_bfill','load_value_corrupted_LI']

input_features = ["load_value_corrupted"] + auxiliary_features  + ["corruption_encoding"]  + encoded_cyclical_features
target_feature =  "load_value"
n_input_features = len(input_features)
print(f"Using following {n_input_features} features as input to data imputation model:{input_features}")

## Specify train and test nodes
selected_train_nodes,selected_test_nodes,selected_eval_nodes = get_train_test_eval_nodes(load_node_dict,train_fraction=0.75,test_fraction=0.2)

## Generate training and testing data
df_train = get_df_node_load_selected_nodes(df_node_load,cyclical_features,selected_train_nodes,corrupted_fraction,multi_corruption=True,consequtive_corruption_probabilities=consequtive_corruption_probabilities,replacement_methods=replacement_methods)
df_test = get_df_node_load_selected_nodes(df_node_load,cyclical_features,selected_test_nodes,corrupted_fraction,multi_corruption=True,consequtive_corruption_probabilities=consequtive_corruption_probabilities,replacement_methods=replacement_methods)
n_train_samples = len(df_train)

## Convert dataframe into a dataset object that can be used by model training
train_input_target = df_to_input_target_dataset(df_train,load_block_length,input_features,target_feature,batch_size,use_prefetch,df_type = "train")
test_input_target = df_to_input_target_dataset(df_test,load_block_length,input_features,target_feature,batch_size,use_prefetch,df_type = "test")

## Create object to normalize data
normalizer = get_normalizer(df_train,input_features,skip_normalization=encoded_cyclical_features+["corruption_encoding"]) #Obtain a normalizer using training data
print(f"Raw data:{test_input_target.take(1).as_numpy_iterator().next()[0][0:2]}")
print(f"Normalized data:{normalizer(test_input_target.take(1).as_numpy_iterator().next()[0][0:2])}")

## Create data imputation model
autoencoder = get_dnn_model(model_type,normalizer,load_block_length,n_input_features)
#autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(),metrics=[tf.keras.metrics.MeanAbsoluteError()]) #tf.keras.losses.MeanAbsoluteError()

monitored_metric = "val_loss"#"val_mean_absolute_error" #"val_loss"
checkpoint_file = 'di_multi_model.epoch{epoch:02d}-loss{val_loss:.5f}' #'-mae{val_mean_absolute_error:.5f}'
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(folder_name_saved_models,f'm-{calendar.month_abbr[selected_month]}_w-{load_block_length}_f-{n_input_features}_c-{corrupted_fraction}_n-{n_train_samples}_{model_type}_'+checkpoint_file),
                             monitor=monitored_metric,
                             verbose=1, 
                             save_best_only=True,
                             mode="min")

callbacks = [checkpoint]

## Train model
tic = time.perf_counter()
history = autoencoder.fit(train_input_target,
                                epochs=n_epochs,
                                shuffle=True,validation_data=test_input_target,callbacks=callbacks)
toc = time.perf_counter()

plot_training_history(history,os.path.join(folder_name_plots,f"loss_history_{distribution_system}_m-{calendar.month_abbr[selected_month]}_w-{load_block_length}_f-{n_input_features}_c-{corrupted_fraction}_n-{n_train_samples}_{model_type}.png"))
autoencoder.summary(expand_nested=True)
print(f"Training took:{toc-tic} s")
num_threads = get_num_cpu_threads()
print(f"Number of CPU threads used for training: {num_threads}")

## Convert best saved model into a 7z file which can be used by the inference script
best_val_loss = min(history.history['val_loss'])
best_epoch = history.history['val_loss'].index(best_val_loss) +1

prediction_model_folder = os.path.join(folder_name_saved_models,f'm-{calendar.month_abbr[selected_month]}_w-{load_block_length}_f-{n_input_features}_c-{corrupted_fraction}_n-{n_train_samples}_{model_type}_di_multi_model.epoch{best_epoch:02d}-loss{best_val_loss:.5f}') 

model_archive = os.path.join(workDir,"app","dataimputation","model",f'di_model-{model_type}_dss-{distribution_system}_m-{calendar.month_abbr[selected_month]}_w-{load_block_length}_f-{n_input_features}_c-{corrupted_fraction}_{model_identifier}')
assert model_type in prediction_model_folder, f"model type:{model_type} not found in {prediction_model_folder}"
model_to_7ziparchive(model_archive,prediction_model_folder)

if config_dict["model_training_details"]["convert_to_tflite"]:
    #tflite_file, _ = os.path.splitext(model_archive)
    saved_model_to_tflite(model_archive,prediction_model_folder)

df_eval = get_df_node_load_selected_nodes(df_node_load,cyclical_features,selected_eval_nodes,corrupted_fraction,multi_corruption=True,consequtive_corruption_probabilities=consequtive_corruption_probabilities,replacement_methods=replacement_methods)

eval_input_target = df_to_input_target_dataset(df_eval,load_block_length,input_features,target_feature,batch_size,use_prefetch,df_type = "eval")

predictions_eval = load_evaluate_predict(prediction_model_folder,input_target=eval_input_target)
df_comparison_eval = compare_performance_moving_window(df_eval,predictions_eval,load_block_length, n_windows = 1400000)
