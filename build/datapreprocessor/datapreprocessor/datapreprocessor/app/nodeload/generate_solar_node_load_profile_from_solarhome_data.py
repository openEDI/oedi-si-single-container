"""'
Created on Friday March 17 11:00:00 2023
@author: Siby Plathottam
"""
import os
import sys
import argparse
import calendar

import pandas as pd
import numpy as np

baseDir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
workDir=os.path.join(baseDir,"datapreprocessor")

print(f"Adding home directory:{baseDir} to path")
sys.path.insert(0,baseDir) #Add module path to prevent import errors

from datapreprocessor.app.nodeload.timeseries_data_utilities import get_n_days_in_df
from datapreprocessor.app.nodeload.nodeload_utilities import check_and_create_folder
from datapreprocessor.app.nodeload.solarnode_utilities import generate_solar_node_profiles

parser=argparse.ArgumentParser()
parser.add_argument('-f','--file',help='Raw solar home data to be used for generating solar node load',default = "solarhome_customers-300_days-365.csv", required=False)
parser.add_argument('-d','--distribution',help='Distribution system',default = "123Bus/case123.dss", required=False)
parser.add_argument('-n','--nsolarnodes',type=int,help='Number of solar nodes for which load shapes are generated',default = 10, required=False)
parser.add_argument('-m','--months',nargs='+', type=int,help='Number of months for which load shapes are generated',default = [7], required=False)
parser.add_argument('-max','--maxsolar',type=float,help='Maximum solar penetration at any solar node',default = 0.3, required=False)

parser.add_argument('-t','--timeperiod',type=str,help='Upsample time period',default = "15Min", required=False)
parser.add_argument('-p','--profilepath',type=str,help='path to save the generated data as profiles',default = "", required=False)
parser.add_argument('--fill',type=int,help='Fill the loadshape by repeating the data to have the given number of points. '+\
	'Works only when profilepath argument is set',default = -1, required=False)
parser.add_argument('--normalize',type=bool,help='When True the returned values are from [0,1]',default = True, required=False)

args=parser.parse_args()

folder_name_timeseries = os.path.join(workDir,"data","solarhome")
folder_name_solarnode = os.path.join(workDir,"data","nodeload")
opendss_casefile = os.path.join(workDir,'data','opendss',args.distribution)
model_folder = args.distribution.split("/")[0]

if not os.path.exists(opendss_casefile):
	raise ValueError(f"{opendss_casefile} is not a valid file!")
else:
	print(f"Found following OpenDSS model from argument:{opendss_casefile}")

check_and_create_folder(folder_name_solarnode)

n_solar_nodes = args.nsolarnodes #10
selected_months = args.months #10
max_solar_penetration  = args.maxsolar #0.3
sample_time_period = args.timeperiod #"15Min"

## Select solar data file for use as base file and generate time series file
df_solar_timeseries = pd.read_csv(os.path.join(folder_name_timeseries,args.file), parse_dates=['datetime'])

## Generate node solar profiles for the distribution system model we are intrested in
df_solar_node,_ = generate_solar_node_profiles(df_solar_timeseries,opendss_casefile,selected_months,n_solar_nodes,max_solar_penetration,sample_time_period=sample_time_period)

## Save solar node profiles
month_names = '-'.join([calendar.month_abbr[num] for num in selected_months])
n_days = get_n_days_in_df(df_solar_node)

if args.normalize:
	for entry in set(df_solar_node.columns).difference(['datetime']):
		df_solar_node[entry]=df_solar_node[entry]/df_solar_node[entry].max()

if not args.profilepath:
	print(f"Saving solar node profiles in solar_node_{model_folder}_{month_names}_nodes-{n_solar_nodes}_days-{n_days}_maxsolar-{max_solar_penetration}.pkl")
	df_solar_node.to_pickle(os.path.join(folder_name_solarnode,\
		f"solar_node_{model_folder}_{month_names}_nodes-{n_solar_nodes}_days-{n_days}_maxsolar-{max_solar_penetration}.pkl"))
else:
	print(f"Saving solar node profiles in {args.profilepath}...")
	cols=[entry for entry in df_solar_node.columns if 'gross_load' in entry]
	for entry in cols:
		if args.fill>0:
			val=df_solar_node[entry].values
			if val.shape[0]>=args.fill:
				val=val[0:args.fill]
			else:
				val=np.tile(val,int(np.ceil(args.fill/val.shape[0])))
				val=val[0:args.fill]
			val.tofile(os.path.join(args.profilepath,'loadshape_'+entry.replace('_gross_load','')+'.csv'),sep='\n')
		else:
			df_solar_node[entry].to_csv(os.path.join(args.profilepath,'loadshape_'+entry.replace('_gross_load','')+'.csv'),index=False)





