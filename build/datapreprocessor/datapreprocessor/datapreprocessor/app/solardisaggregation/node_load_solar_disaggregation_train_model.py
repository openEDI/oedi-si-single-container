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

from oedianl.app.nodeload.smartmeter_data_preprocessing import valid_load_types_v2,get_time_series_dataframe,valid_load_types
from oedianl.app.nodeload.timeseries_data_utilities import get_config_dict
from oedianl.app.nodeload.nodeload_preprocessing import encode_cyclical_features
from oedianl.app.nodeload.datapipeline_utilities import get_input_target_dataset,check_moving_window,get_train_test_eval_nodes,df_to_input_target_dataset
from oedianl.app.nodeload.nodeload_utilities import check_and_create_folder,get_upsampled_df
from oedianl.app.dataimputation.dae_models import get_dnn_model,get_normalizer,plot_training_history,model_to_7ziparchive
from oedianl.app.solardisaggregation.solardisaggregation_preprocessing import convert_solardata_to_timeseries,generate_solar_node_profiles,get_df_train_solar_disaggregation

rng = default_rng()

#folder_name_raw = os.path.join(workDir,"data","solardisaggregation")
folder_name_timeseries = os.path.join(workDir,"data","solarhome")
folder_name_plots = os.path.join(workDir,"app","solardisaggregation","plots")

parser=argparse.ArgumentParser()
parser.add_argument('-c','--config',help='config to be passed to the solar disaggregation training script',default = "solar_disaggregation_config.json", required=False)
args=parser.parse_args()
config_file = args.config

config_file = os.path.join(workDir,"app","solardisaggregation",config_file)
config_dict= get_config_dict(config_file)

selected_timeseries_file  = config_dict["nodeload_data_details"]["selected_timeseries_file"] #Specify file containing time series solar home data
selected_months = config_dict["nodeload_data_details"]["selected_months"] #[7] # The months for which we are developing the model
distribution_system = config_dict["nodeload_data_details"]["distribution_system"] #"123Bus"
distribution_system_file = config_dict["nodeload_data_details"]["distribution_system_file"] #"123Bus"
upsample_original_time_series = config_dict["nodeload_data_details"]["upsample_original_time_series"] # True
upsample_time_period = config_dict["nodeload_data_details"]["upsample_time_period"] #"15Min"
n_days = config_dict["nodeload_data_details"]["n_days"] #31
n_nodes = config_dict["train_data_details"]["n_nodes"] #4 #The number of node load profiles that will be used for training
n_customers = config_dict["nodeload_data_details"]["n_customers"] #300

max_solar_penetration = config_dict["nodeload_data_details"]["max_solar_penetration"] #0.3 #The maximum solar penetration at a node

load_block_length = config_dict["train_data_details"]["load_block_length"] #4 #The lenghth of the time window

model_type = config_dict["model_arch_details"]["model_type"] #"lstm" #"1dcnn"#"lstm" #Currently enther lstm or 1dcnn
stateful = False #True #False

use_prefetch= True
batch_size =  config_dict["model_training_details"]["batch_size"] #32
n_epochs =  config_dict["model_training_details"]["n_epochs"] #5
model_identifier = config_dict["model_training_details"]["model_identifier"]  #"v0"

opendss_casefile = os.path.join(workDir,"data","opendss",distribution_system_file)
month_names = '-'.join([calendar.month_abbr[num] for num in selected_months])
folder_name_saved_models = os.path.join(workDir,"app","solardisaggregation","saved_models",distribution_system,f'month-{month_names}')

check_and_create_folder(folder_name_plots)
check_and_create_folder(folder_name_saved_models)

## Generated base solar profiles from solar data
df_solar_timeseries = pd.read_csv(os.path.join(folder_name_timeseries,selected_timeseries_file), parse_dates=['datetime'])

## Generate node solar profiles for the distribution system model we are intrested in
df_solar_node,solar_node_dict = generate_solar_node_profiles(df_solar_timeseries,opendss_casefile,selected_months,n_solar_nodes=n_nodes,max_solar_penetration = max_solar_penetration,upsample_time_series=upsample_original_time_series,upsample_time_period=upsample_time_period)

## Select features for model training
cyclical_features = ["hour_of_day",'day_of_week','weekend']
encoded_cyclical_features= ['cos_hour','sin_hour','cos_day_of_week','sin_day_of_week']#,'weekend']

input_features = ["net_load"] + encoded_cyclical_features #gross_load
target_feature =  "solar_power"
target_features = ["solar_power"]
n_input_features = len(input_features)
print(f"Using following {n_input_features} features as input to data imputation model:{input_features}")

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
