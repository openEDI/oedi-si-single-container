"""'
Created on Thursday Feb 27 15:00:00 2023
@author: Siby Plathottam
"""
import os
import sys
import argparse
import calendar

from numpy.random import default_rng

baseDir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  #Add path of home directory e.g.'/home/splathottam/GitHub/oedi'
workDir=os.path.join(baseDir,"datapreprocessor")

print(f"Adding home directory:{baseDir} to path")
sys.path.insert(0,baseDir) #Add module path to prevent import errors

from datapreprocessor.app.nodeload.timeseries_data_utilities import get_config_dict
from datapreprocessor.app.nodeload.nodeload_utilities import create_average_timeseries_profiles,generate_load_node_profiles,check_and_create_folder
from datapreprocessor.app.nodeload.datapipeline_utilities import get_train_test_eval_nodes,df_to_input_target_dataset
from datapreprocessor.app.dataimputation.data_imputation_preprocessing import get_df_node_load_selected_nodes
from datapreprocessor.app.model_utilities.model_utilities import get_autoencoder_model,get_compiled_model,get_checkpoint_callback,get_normalizer,evaluate_predict,check_normalizer
from datapreprocessor.app.model_utilities.model_training_utilities import train_model,get_best_model
from datapreprocessor.app.model_utilities.model_save_load_utilities import model_to_archive,modelarchive_to_modelpath,load_keras_model

rng = default_rng()

## Specify locations of plots and models
folder_plots = os.path.join(baseDir,"datapreprocessor","app","anomalydetection","plots")
folder_model_inference = os.path.join(baseDir,"datapreprocessor","app","anomalydetection","model")
folder_model_archive = os.path.join(baseDir,"datapreprocessor","app","anomalydetection","model_archives")
folder_model_checkpoints = os.path.join(baseDir,"datapreprocessor","app","anomalydetection","model_checkpoints")
check_and_create_folder(folder_plots)
check_and_create_folder(folder_model_inference)
check_and_create_folder(folder_model_archive)
check_and_create_folder(folder_model_checkpoints)

parser=argparse.ArgumentParser()
parser.add_argument('-c','--config',help='config to be passed to the anomaly detection  training script',default = "anomaly_detection_config.json", required=False)
args=parser.parse_args()
config_file = args.config

config_file = os.path.join(workDir,"app","anomalydetection",config_file)
config_dict= get_config_dict(config_file)

## Select timeseries file for use as base file
selected_timeseries_files  = config_dict["nodeload_data_details"]["selected_timeseries_files"] #Specify file containing zip code level time series data from individual smart meters
selected_timeseries_files = [os.path.join(workDir,timeseries_file) for timeseries_file in selected_timeseries_files]

## Specify details of anonymized node load profiles
upsample_original_time_series = config_dict["nodeload_data_details"]["upsample_original_time_series"] # Should the original time series be upsampled
upsample_time_period = config_dict["nodeload_data_details"]["upsample_time_period"] #"Time period of upsampling
selected_month = config_dict["nodeload_data_details"]["selected_month"] #2 # The month for which we are developing the model
distribution_system = config_dict["nodeload_data_details"]["distribution_system"] ##The distribution system we are generating the profiles
distribution_system_file = config_dict["nodeload_data_details"]["distribution_system_file"] ##The opendss file
measurement_column = config_dict["nodeload_data_details"]["measurement_column"]
opendss_casefile = os.path.join(baseDir,"datapreprocessor","data",distribution_system_file)
n_days = config_dict["train_data_details"]["n_days"] #4 #The number of full day profiles that will be generated for training
n_nodes = config_dict["train_data_details"]["n_nodes"] #4 #The number of nodes that will be used for training
load_scaling_mode = config_dict["nodeload_data_details"]["load_scaling_mode"] #"simple" #multi

## Training data specifications
cyclical_features = config_dict["train_data_details"]["cyclical_features"] #["hour_of_day",'day_of_week','weekend'] #Cyclical features to be added to the training data

## Input/target features for anomaly detection model
encoded_cyclical_features= ['cos_hour','sin_hour']#,'cos_day_of_week','sin_day_of_week','weekend']
auxiliary_features = []#,'load_value_corrupted_bfill','load_value_corrupted_LI']
input_features = [f"{measurement_column}"] + auxiliary_features + encoded_cyclical_features
target_feature =  f"{measurement_column}"
n_input_features = len(input_features)
n_target_features =  1
print(f"Using following {n_input_features} features as input to anomaly detection model:{input_features}")

## Anomaly detection model architecture details
stateful = False #True #False
window_size =  config_dict["train_data_details"]["window_size"] #4 #The length of the time window
model_type = config_dict["model_arch_details"]["model_type"] #"lstm" #"1dcnn"#"lstm" #Currently enther lstm or 1dcnn

## Anomaly detection model training details
use_prefetch= True
batch_size =  config_dict["model_training_details"]["batch_size"] #32
n_epochs =  config_dict["model_training_details"]["n_epochs"] #5
monitored_metric = "val_loss"#"val_mean_absolute_error" # Performance metric monitored during training
model_identifier = config_dict["model_training_details"]["model_identifier"]  #"v0"

## Generated averaged load profiles for all load type within the selected time series file
df_averaged_load,df_averaged_day_load = create_average_timeseries_profiles(timeseries_files=selected_timeseries_files,month=selected_month,convert_to_kW=True,upsample=upsample_original_time_series,upsample_time_period=upsample_time_period)
## Generate anonymized node load profiles for the selected distribution system model
df_node_load,load_node_dict = generate_load_node_profiles(df_averaged_day_load,case_file=opendss_casefile,n_nodes=n_nodes,n_days=n_days,start_year = 2016,start_month=selected_month,start_day=1,scaling_type=load_scaling_mode)

## Specify train, test and eval nodes
selected_train_nodes,selected_test_nodes,selected_eval_nodes = get_train_test_eval_nodes(load_node_dict,train_fraction=0.75,test_fraction=0.2)

## Generate training, testing, and evaluation data
df_train = get_df_node_load_selected_nodes(df_node_load,cyclical_features,selected_train_nodes,measurement_column,corrupted_fraction=0.0) #We use same method as data imputation but with no data corruption
df_test = get_df_node_load_selected_nodes(df_node_load,cyclical_features,selected_test_nodes,measurement_column,corrupted_fraction=0.0) #We use same method as data imputation but with no data corruption
df_eval = get_df_node_load_selected_nodes(df_node_load,cyclical_features,selected_eval_nodes,measurement_column,corrupted_fraction=0.0) #We use same method as data imputation but with no data corruption
n_train_samples = len(df_train)

## Convert dataframe into a dataset object that can be used by model training
input_target_dataset_train = df_to_input_target_dataset(df_train,window_size,input_features,target_feature,batch_size,use_prefetch=True,df_type = "train")
input_target_dataset_test = df_to_input_target_dataset(df_test,window_size,input_features,target_feature,batch_size,use_prefetch=True,df_type = "test")
input_target_dataset_eval = df_to_input_target_dataset(df_eval,window_size,input_features,target_feature,batch_size,use_prefetch=True,df_type = "eval")

## Create object to normalize data
normalizer = get_normalizer(df_train,input_features,skip_normalization=encoded_cyclical_features+["corruption_encoding"]) #Obtain a normalizer using training data
check_normalizer(normalizer,input_target_dataset_test)

## Create anomaly detection model
ad_model = get_autoencoder_model(model_type,window_size,n_input_features,n_target_features,normalizer=normalizer)
ad_model = get_compiled_model(ad_model)

## Create checkpoints
model_id = f'ad_model-nodeload-{distribution_system}_m-{calendar.month_abbr[selected_month]}_w-{window_size}_f-{n_input_features}-{model_type}'
model_checkpoint_id = 'epoch{epoch:02d}-loss{val_loss:.5f}' #'-mae{val_mean_absolute_error:.5f}'
model_checkpoint_path=os.path.join(folder_model_checkpoints,f'{model_id}_n-{n_train_samples}_{model_checkpoint_id}')
callbacks = [get_checkpoint_callback(model_checkpoint_path,monitored_metric,save_weights_only=False)]

## Train model
di_model,history = train_model(ad_model,input_target_dataset_train,input_target_dataset_test,n_epochs,callbacks) #note that is method returns the last model

## Find best model checkpoint
best_monitored_metric,best_epoch = get_best_model(history,monitored_metric)
best_checkpoint_id = f'epoch{best_epoch:02d}-loss{best_monitored_metric:.5f}.keras'
best_model_savepath = os.path.join(folder_model_checkpoints,f'{model_id}_n-{n_train_samples}_{best_checkpoint_id}')
print(f"Best model checkpoint:{best_model_savepath}")

## Save best model in model archive for inference
best_model_archivepath = model_to_archive(best_model_savepath,os.path.join(folder_model_archive,f'{model_id}_{model_identifier}'))

## Load and Evaluate pre-trained model
best_model_savepath = modelarchive_to_modelpath(model_archivepath = best_model_archivepath,model_folder = folder_model_inference)
best_model = load_keras_model(best_model_savepath)
predictions_eval = evaluate_predict(best_model,input_target=input_target_dataset_eval)
