# Introduction
The Data preprocessing application contains the following applications:

1. **Data Imputation**: Takes net load at any distribution system node as input. If missing data is encountered at any time stamp at any node, the federate returns an estimated load for that time stamp.  The application source code can be found [here](https://github.com/openEDI/oedi-si-single-container/tree/main/build/datapreprocessor/datapreprocessor/datapreprocessor/app/dataimputation). New data imputation models can be trained using: ```python app/dataimputation/node_load_data_imputation_train_model.py -c "data_imputation_config.json" ```

   The trained data imputation model can be used within the data imputation [federate](https://github.com/openEDI/oedi-si-single-container/tree/main/build/datapreprocessor/datapreprocessor/datapreprocessor/federates/dataimputation).

2. **Solar Disaggregation**: Takes net load at distribution system solar node as input and outputs the estimated solar power produced. The application source code can be found [here](https://github.com/openEDI/oedi-si-single-container/tree/main/build/datapreprocessor/datapreprocessor/datapreprocessor/app/solardisaggregation).
    New solar disaggregation models can be trained using: ```python node_load_solar_disaggregation_train_model.py -c "solar_disaggregation_config.json" ```


3. **Anomaly Detection**: Takes net load at any distribution system node as input. If anomalous data is encountered at any time stamp, a flag that tags the data at the time stamp as an anomaly is set to true.  New anomaly detection models can be trained using ```python node_load_anomaly_detection_train_model.py -c "anomaly_detection_config.json" ```

4. **Synthetic load and solar node data**: This application can be used to generate synthetic time series load data for any distribution system using historical smart meter data or solar home data. The application source code can be found [here](https://github.com/openEDI/oedi-si-single-container/tree/main/build/datapreprocessor/datapreprocessor/datapreprocessor/app/nodeload). New data can be generated using: 
- Using smart meter data: ```python generate_load_node_load_profile_from_smartmeter_data.py -f "smartmeter_timeseries.csv" -id "smartmeter" -d "123Bus/case123.dss" -n 10```
- Using solar home data: ```python /generate_solar_node_load_profile_from_solarhome_data.py -f "solarhome_customers-50_days-10.csv" -id "solarhome" -d "123Bus/case123.dss" -n 10```

## Deploying Data Imputation application within a co-simulation

The Data Imputation application can be deployed in OEDI using the following steps:

1. Identify a source of data. Currently only AMI, solarhome, and pmu data is supported.
2. Identify a distribution system. Currently only OpenDSS models are supported.
3. Populate a configuration file. Details of JSON file are described [below](#trainconfig)
4. Train a new data imputation model can be trained using:
```python
python node_load_data_imputation_train_model.py -c "data_imputation_config.json"
```
5. Provide details of the model to the data imputation federate through the *static_inputs.json* file. Details of JSON file are described  [below](#federateconfig).
6. Now the federate can be used just like any other federate by adding it to the *runner_config*.json file.

### Description of model training JSON configuration file{#trainconfig}

An example the data_imputation_config.json that must be supplied by the user is shown below. 

```json
{
  "nodeload_data_details": {
    "selected_timeseries_files": ["solarhome/solarhome_customers-300_days-365.csv"],
    "selected_month": 2,
    "distribution_system": "smartds_no_load_shape",
    "distribution_system_file": "smartds_no_load_shape/Master.dss",
    "load_scaling_type": "simple",
    "upsample_original_time_series": true,
    "upsample_time_period": "15Min"
  },

  "train_data_details": {
    "n_days": 4,
    "n_nodes": 100,
    "load_block_length": 4
  },

  "model_arch_details": {
    "model_type": "lstm"
  },

  "model_training_details": {
    "n_epochs": 2,
    "batch_size": 32,
    "model_identifier": "v0"
  }
}
```

The above JSON file has four main sections, namely:
- nodeload_data_details
- train_data_details
- model_arch_details
- model_training_details

#### nodeload_data_details
This section provides the details for the data used in the model, such as the selected time-series files, the month that is being used, the distribution system, the distribution system file, the load scaling type, and whether to upsample the original time-series data.

- selected_timeseries_files: This is an array of strings that lists the names of the CSV files containing the time-series data.
- selected_month: This is an integer value that represents the month for which the data is being used.
- distribution_system: This is a string that identifies the distribution system being used. It will be used to name the model.
- distribution_system_file: This is a string that provides the path to the distribution system file being used.
- load_scaling_type: This is a string that specifies the type of load scaling being used. Current options:"simple", "multi"
- upsample_original_time_series: This is a boolean value that indicates whether the original time-series data needs to be upsampled.
- upsample_time_period: This is a string that specifies the time period for the upsampled data. E.g. "15Min", "30Min"

#### train_data_details
This section provides the details for the training data, such as the number of days being used, the number of nodes, and the load block length.

- n_days: This is an integer value that represents the number of days of data being used for training.
- n_nodes: This is an integer value that represents the number of distribuion nodes in the training data.
- load_block_length: This is an integer value that represents the length of the look back window.

#### model_arch_details
This section provides the details for the model architecture, such as the type of model being used.

- model_type: This is a string that specifies the type of model being used. Current options: "lstm", "1Dcnn"

#### model_training_details
This section provides the details for the model training, such as the number of epochs, batch size, and the model identifier.

- n_epochs: This is an integer value that represents the number of epochs for training the model.
- batch_size: This is an integer value that represents the batch size used during training.
- model_identifier: This is a string that provides an identifier for the model.

### Configuration File for Data Imputation federate{#federateconfig}

This JSON configuration file is required for using a data imputation model with a data imputation federate. The fields in the file are described below:

```
{
    "modelDir": "app/dataimputation/model",
    "pretrained_model_file": "di_model-lstm_dss-123Bus_m-Feb_w-4_f-5_c-0.05_v0.7z",
    "monitored_nodes": ["s100c", "s102c"],
    "initial_measurements": {"s100c":10.0, "s102c":10.0},
    "window_size": 4,
    "input_features": ["load_value_corrupted", "load_value_corrupted_ffill", "corruption_encoding", "cos_hour", "sin_hour"]
}
```

The fields are described below:
-  modelDir: Specifies the directory path where the model is saved or will be saved.
- pretrained_model_file: Specifies the name of the pretrained model file. This model file is used for data imputation.
- monitored_nodes: Specifies the list of distribution nodes to be monitored for data imputation.
- initial_measurements: Specifies the initial measurement values for the nodes specified in the monitored_nodes field.
- window_size: Specifies the size of the context window for the input data. 
- input_features: Specifies the list of input features for the model. These features are used as input to the data imputation model. The features include:
        * load_value_corrupted: the corrupted load values
            * load_value_corrupted_ffill: the corrupted load values with forward filling
            * corruption_encoding: a binary encoding of the corruption status of the load values
            * cos_hour: the cosine of the hour of the day
            * sin_hour: the sine of the hour of the day. The `cos_hour` and `sin_hour` features are used to capture the cyclic behavior of the load values over time.