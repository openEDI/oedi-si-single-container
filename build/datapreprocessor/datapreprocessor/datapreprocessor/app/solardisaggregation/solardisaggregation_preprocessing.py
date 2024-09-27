"""'
Created on Wed November 30 10:00:00 2022
@author: Siby Plathottam
"""

from typing import List, Set, Dict, Tuple, Optional, Union

import pandas as pd
from tqdm import tqdm

from datapreprocessor.app.nodeload.nodeload_preprocessing import encode_cyclical_features

def get_df_train_solar_disaggregation(df_solar_node,cyclical_features,selected_nodes):
	"""Create training data"""
	
	selected_nodes.sort()
	print(f"Selected {len(selected_nodes)} samples:{selected_nodes}")
	n_timesteps = len(df_solar_node)
	print(f"Following loads with {n_timesteps} time steps were selected:{selected_nodes}")
	
	df_train = pd.DataFrame()
	
	node_solar_power_values = []
	node_gross_load_values = []
	node_net_load_values = []
	node_solar_ids = []
	node_solar_time_stamps = []	   
	
	for node_id in tqdm(selected_nodes):
		node_solar_power_values.extend(df_solar_node[f"{node_id}_solar"].values)
		node_gross_load_values.extend(df_solar_node[f"{node_id}_gross_load"].values)
		node_net_load_values.extend(df_solar_node[f"{node_id}_net_load"].values)
		node_solar_time_stamps.extend(list(df_solar_node["datetime"].values))
		node_solar_ids.extend([node_id]*n_timesteps)
		
	df_train["datetime"] = node_solar_time_stamps
	df_train["solar_power"] = node_solar_power_values
	df_train["gross_load"] = node_gross_load_values
	df_train["net_load"] = node_net_load_values
	df_train["node_id"] = node_solar_ids
	
	df_train = encode_cyclical_features(df_train,cyclical_features,show_df=False,show_plot=False)
	
	return df_train
