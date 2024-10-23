"""
Created on Wed May 01 00:10:00 2018
@author: splathottam
"""

import os
import glob
import random
import warnings

try:
	import py7zr
except ImportError:
	warnings.warn('py7zr failed to import', ImportWarning)

def extract_7z_files(source_folder,delete_after_extraction=False):
	"""Extract 7z files to zip files in the same folder"""
	
	zipfiles_7z = glob.glob(os.path.join(source_folder,'*.7z') ) #List only zip files
	print(f"{len(zipfiles_7z)} 7z files found in {source_folder}:{zipfiles_7z}")
	
	for zipfile_7z in zipfiles_7z:
		print(f"Extracting {zipfile_7z}...")
		with py7zr.SevenZipFile(zipfile_7z, 'r') as archive:
			archive.extractall(path=source_folder)
		if delete_after_extraction:
			print(f"Removing zip file:{zipfile_7z}")
			os.remove(zipfile_7z)
			
def get_n_timeseries_files(file_names_time_series,n=1,month=1):
	"""Randomly select n files"""
	
	filtered_timeseries_files = [file_name_time_series for file_name_time_series in file_names_time_series if f"_{str(month).rjust(2, '0')}_" in file_name_time_series]
	print(f"Found {len(filtered_timeseries_files)} files for month:{month}")
	#print(filtered_timeseries_files)
	if not len(filtered_timeseries_files) >= n:
		print(f"Number of files available:{len(filtered_timeseries_files)} is less than required:{n}")
		n = len(filtered_timeseries_files)
		
	return random.sample(filtered_timeseries_files, n)
