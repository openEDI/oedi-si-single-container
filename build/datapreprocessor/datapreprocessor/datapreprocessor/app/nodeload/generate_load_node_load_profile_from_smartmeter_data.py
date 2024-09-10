"""
Created on Mon September 12 00:12:00 2022
@author: Siby Plathottam
"""

import os
import sys
import argparse
import calendar

baseDir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) Add path of home directory 
workDir=os.path.join(baseDir,"datapreprocessor")

print(f"Adding home directory:{baseDir} to path")
sys.path.insert(0,baseDir) #Add module path to prevent import errors

from datapreprocessor.app.nodeload.nodeload_utilities import create_average_timeseries_profiles,generate_load_node_profiles,check_and_create_folder

parser=argparse.ArgumentParser()
parser.add_argument('-f','--timeseries',help='Time series files to be used for generating node load',default = "solarhome/solarhome_customers-300_days-365.csv", required=False)
parser.add_argument('-id','--timeseriesid',help='Identification for time series file',default = "solarhome_aus", required=False)
parser.add_argument('-d','--distribution',help='Distribution system',default = "123Bus/case123.dss", required=False)
parser.add_argument('-m','--month',type=int,help='Month for which load shapes are generated',default = 2, required=False)
parser.add_argument('-n','--ndays', type=int,help='Number of days',default = 10, required=False)
parser.add_argument('-u','--upsample',type=bool,help='Whether to upsample',default = True, required=False)
parser.add_argument('-t','--timeperiod',help='Upsample time period',default = "15Min", required=False)

args=parser.parse_args()

folder_name_timeseries = os.path.join(workDir,"data")
folder_name_nodeload = os.path.join(workDir,"data","nodeload")
timeseries_file = os.path.join(folder_name_timeseries,args.timeseries)
opendss_casefile = os.path.join(workDir,'data','opendss',args.distribution)
model_folder = args.distribution.split("/")[0]

if not os.path.exists(timeseries_file):
    raise ValueError(f"{timeseries_file} is not a valid file!")
else:
	print(f"Found following time series file from argument:{timeseries_file}")
	timeseries_files = [timeseries_file]

if not os.path.exists(opendss_casefile):
	raise ValueError(f"{opendss_casefile} is not a valid file!")
else:
	print(f"Found following OpenDSS model from argument:{opendss_casefile}")

check_and_create_folder(folder_name_nodeload)

upsample_original_time_series = args.upsample #True
upsample_time_period = args.timeperiod #"15Min"
selected_month = args.month #10
n_days = args.ndays #10

df_averaged_load,df_averaged_day_load = create_average_timeseries_profiles(timeseries_files=timeseries_files,month=selected_month,convert_to_kW=True,upsample=upsample_original_time_series,upsample_time_period=upsample_time_period)

df_node_load,load_node_dict = generate_load_node_profiles(df_averaged_day_load,case_file=opendss_casefile,n_days=n_days)
month = list(set(df_averaged_day_load["datetime"].dt.month))[0] #Find month from time series - assume data from only one month is present

csv_file_name = f"smartmeter_averaged_day_load_m-{calendar.month_abbr[month]}.csv"
print(f"Saving smart meter averaged day load in {csv_file_name}")
df_averaged_day_load.to_csv(os.path.join(folder_name_nodeload,f"smartmeter_averaged_day_load_m-{calendar.month_abbr[month]}.csv"),index=False)
csv_file_name = f"load_node_model-{model_folder}_timeseries-{args.timeseriesid}_m-{calendar.month_abbr[month]}_days-{n_days}.csv"
print(f"Saving node load data in {csv_file_name}")
df_node_load.to_csv(os.path.join(folder_name_nodeload,csv_file_name),index=False)
