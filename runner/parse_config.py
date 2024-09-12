import os
import json
import copy
import pdb
import argparse


class Build(object):

	def __init__(self,userConfigPath=None):
		self._baseDir=os.path.dirname(os.path.abspath(__file__))
		self.config={}
		self.config['federate_template_config']=json.load(open(\
			os.path.join(self._baseDir,'federate_template_config.json')))
		self.config['availableFederates']=json.load(open(os.path.join(self._baseDir,'available_federates.json')))

		if not userConfigPath:
			userConfigPath=os.path.join(self._baseDir,'user_config.json')
		print(f"Opening user configuration from:{userConfigPath}")
		self.config['userConfig']=json.load(open(userConfigPath))

		self.config['links']=self._convertToOedisiLinks(json.load(open(os.path.join(self._baseDir,'links.json'))))
		self.config['oedisi_runtime_federates']=['feeder','sensor_voltage_real','sensor_voltage_imaginary',\
			'sensor_power_real','sensor_power_imaginary','recorder_voltage_real','recorder_voltage_imag']
		self.config['oedisi_runtime_wiring_diagram']=json.load(open(os.path.join(self._baseDir,'oedisi_runtime_wiring_diagram.json')))
		self.userFederateOptions={'allowedApplicationTypes':['dsse','dopf'],'supportedLanguages':['python']}

	def _get_template_federate_config(self,name,type,directory,exec,hostname='localhost',parameters=None,src=None):
		conf=copy.deepcopy(self.config['federate_template_config'])
		conf['wiringDiagramData']['components']['name']=name
		conf['wiringDiagramData']['components']['type']=type
		conf['configRunnerData']['federates']['name']=name
		conf['configRunnerData']['federates']['directory']=directory
		conf['configRunnerData']['federates']['exec']=exec
		conf['configRunnerData']['federates']['hostname']=hostname

		if parameters:
			conf['wiringDiagramData']['components']['parameters']=parameters

		if src:
			conf['data']['src']=src

		return conf

	def _update_wiring_diagram_data(self,wiringDiagramData,conf):
		wiringDiagramData['components'].append(conf['wiringDiagramData']['components'])
		wiringDiagramData['links'].extend(conf['wiringDiagramData']['links'])

	def _convertToOedisiLinks(self,data):
		res=[]
		for sourceData,targetData in data:
			res.append({"source":sourceData[0],"source_port":sourceData[1],\
				"target":targetData[0],"target_port":targetData[1]})
		return res


	def write_config(self):
		userConfig=self.config['userConfig']
		availableFederates=self.config['availableFederates']
		simulationConfig=userConfig['simulation_config']
		
		userConfig['federates']=[]
		stateEstimatorWiringData=json.load(open(os.path.join(self._baseDir,'oedisi_state_estimator_wiring_diagram.json')))

		# check if use_oedisi_runtime is set to true
		if userConfig['use_oedisi_runtime']:
			userConfig['federates'].extend(self.config['oedisi_runtime_federates'])
			userConfig['federates']=list(set(userConfig['federates']))

		if userConfig['oedisi_runtime_federates']:
			userConfig['federates'].extend(userConfig['oedisi_runtime_federates'])
			userConfig['federates']=list(set(userConfig['federates']))

		# user provided federate mods
		if userConfig['user_provided_federates']:
			userConfig['federates'].extend([entry['name'] for entry in userConfig['user_provided_federates']])
			userConfig['federates']=list(set(userConfig['federates']))
			for thisFederate in userConfig['user_provided_federates']:
				assert thisFederate['application_type'].lower() in self.userFederateOptions['allowedApplicationTypes'],\
					"Only the following application types are supported for user defined federates,"+\
					f"{self.userFederateOptions['allowedApplicationTypes']} but got {thisFederate['application_type'].lower()} in config"
				assert thisFederate['language'].lower() in self.userFederateOptions['supportedLanguages'],\
					"Only the following languages are supported for user defined federates,"+\
					f"{self.userFederateOptions['supportedLanguages']} but got {thisFederate['language'].lower()} in config"

				availableFederates[thisFederate['name']]={"directory":thisFederate['name'],"name":thisFederate['name'],\
					"executable":thisFederate['executable'],"hostname": "localhost"}
				# copy template for wiring diagram
				if thisFederate['application_type'].lower()=='dsse':
					stateEstimatorWiringData[thisFederate['name']]=copy.copy(stateEstimatorWiringData['state_estimator_nrel'])
					stateEstimatorWiringData[thisFederate['name']]=json.loads(json.dumps(\
						stateEstimatorWiringData[thisFederate['name']]).replace('state_estimator_nrel',\
						thisFederate['name']).replace('_nrel','_'+thisFederate['name']))
					if not 'parameters' in thisFederate:
						thisFederate['parameters']={}
					stateEstimatorWiringData[thisFederate['name']]['components'][0]['parameters']=thisFederate['parameters']

		# check federate requirement
		print(f"Available federates:{list(availableFederates.keys())}")
		unavailableFed=set(userConfig['federates']).difference(availableFederates)
		#assert not unavailableFed,f"federates {unavailableFed} are unavailable"

		# generate wiring diagram
		appFederates=list(set(userConfig['federates']).difference(self.config['oedisi_runtime_federates']))
		print(f"Following application federates will be added:{appFederates}")

		wiringDiagramData={'name':userConfig['name'],'components':[],'links':[]}
		# add components even if userConfig['use_oedisi_runtime'] is False and remove later as needed
		temp=copy.deepcopy(self.config['oedisi_runtime_wiring_diagram'])

		for n in range(len(temp['components'])):
			if temp['components'][n]['name']=='feeder':
				temp['components'][n]['parameters'].update(simulationConfig)

		wiringDiagramData['components'].extend(temp['components'])
		wiringDiagramData['links'].extend(temp['links'])

		#for thisAppFederate in appFederates: #Not using this anymore
			#assert thisAppFederate in stateEstimatorWiringData,f'{thisAppFederate} not in stateEstimatorWiringData'
			#print(f"Adding {thisAppFederate} to wiring diagram...")
			#wiringDiagramData['components'].extend(stateEstimatorWiringData[thisAppFederate]['components'])
			#wiringDiagramData['links'].extend(stateEstimatorWiringData[thisAppFederate]['links'])

		if not os.path.exists('/home/run'):
			print("Creating directory /home/run...")
			os.system('mkdir /home/run')
		# make modifciations for datapreprocessor
		if self.config['userConfig']['use_oedisi_preprocessor']:			
			preprocessorFederates=build.config['userConfig']['oedisi_preprocessor_federates']
			preprocessorFederatesDir='/home/datapreprocessor/datapreprocessor/federates'
			availablePreprocessorFederates=os.listdir(preprocessorFederatesDir) #Check datapreprocessor folder for available federates
			print(f"Following preprocessor federates are available:{availablePreprocessorFederates}")
			for thisFederate in preprocessorFederates:
				thisFederate=thisFederate.replace('-','').replace('_','') #### TODO: foo_bar to foobar
				if thisFederate in availablePreprocessorFederates:					
					flag=os.system(f'cp -r {os.path.join(preprocessorFederatesDir,thisFederate)} /home/run') #copy datapreprocessor federates to /home/run
					assert flag==0, f'copying {thisFederate} failed with error flag={flag}'

		# template based federate additions to wiring diagram (dopf_ornl)
		#### TODO: Need to make changes to config data structure to include dopf federates
		"""
		thisConf=self._get_template_federate_config('dopf_ornl','OptimalPowerFlowORNL',
			'/home/run/dopf_ornl','python /home/run/dopf_ornl/opf_federate.py',src='/home/oedisi-dopf-ornl/dopf_federate')
		self._update_wiring_diagram_data(wiringDiagramData,thisConf)
		"""
		
		# write wiring diagram and generate config_runner #not using wiring_diagram
		#wiring_diagram_path=os.path.join(self._baseDir,'wiring_diagram.json')
		#json.dump(wiringDiagramData,open(wiring_diagram_path,'w'),indent=3)
				
		components_links_library_path = '/home/runtime/runner/components_links_library.json' #JSON file containing components and links of application federates
		system_json_path = '/home/runtime/runner/system_custom.json' #'/home/runtime/runner/system.json' 
		components_json_path = '/home/runtime/runner/components.json' #works
		removeFederates = []
		#Update paths in system_json
		with open(system_json_path, "r") as file:
			system_json = json.load(file)
		with open(components_links_library_path, "r") as file:
			components_links_library = json.load(file)

		for appFederate in appFederates: #Loop through appFederates and add components and links to system_json
			if appFederate in ["state_estimator_nrel","state_estimator_pnnl","dopf_nrel","dopf_ornl","dopf_pnnl"]:
				print(f"Adding {appFederate} to system_json...")
				system_json["components"].extend(components_links_library[appFederate]["components"])
				system_json["links"].extend(components_links_library[appFederate]["links"])
		
		for component in system_json["components"]:
			if "feather_filename" in component["parameters"]:				
				component["parameters"]["feather_filename"] = component["parameters"]["feather_filename"].replace("../../", "/home/") #assign the result of replace back to the dictionary key.
			if "csv_filename" in component["parameters"]:				
				component["parameters"]["csv_filename"] = component["parameters"]["csv_filename"].replace("../../", "/home/") #assign the result of replace back to the dictionary key
			if "topology_output" in component["parameters"]:				
				component["parameters"]["topology_output"] = component["parameters"]["topology_output"].replace("../../", "/home/") #assign the result of replace back to the dictionary key.
		
		print(f"Following federates are present:{[item['name'] for item in system_json['components'] if not any(keyword in str(value).lower() for value in item.values() for keyword in removeFederates)]}")
		system_json["components"] = [item for item in system_json["components"] if not any(keyword in str(value).lower() for value in item.values() for keyword in removeFederates)]
		system_json["links"] = [item for item in system_json["links"] if not any(keyword in str(value).lower() for value in item.values() for keyword in removeFederates)]

		with open(system_json_path, "w") as file: # Save the updated JSON back to the file
			json.dump(system_json, file, indent=4)
		
		#Create system_runner.json from components.json and system.json
		directive=f'oedisi build --target-directory /home/run --component-dict {components_json_path} --system {system_json_path}'		
		print(f"Executing directive:{directive}")
		flag=os.system(directive)
		assert flag==0,f'generating config_runner failed with flag:{flag}'
		
		"""
		# template based federate additions (dopf_ornl)
		if thisConf['data']['src']:# replace template data after wiring diagram puts default data
			os.system(f'rm -r /home/run/{thisConf["configRunnerData"]["federates"]["name"]}/* && '+\
				f'cp -r {thisConf["data"]["src"]}/* /home/run/{thisConf["configRunnerData"]["federates"]["name"]}')
		"""
		
		config_runner=json.load(open('/home/run/system_runner.json')) #Load system_runner.json into config_runner
		
		""" #we don't need this
		# config_runner modifications for dopf_ornl
		ind=-1
		for n in range(len(config_runner['federates'])):
			if config_runner['federates'][n]['name']==thisConf['configRunnerData']['federates']['name']:
				ind=n
		if ind>=0:
			config_runner['federates'].pop(ind)
			config_runner['federates'].append(thisConf['configRunnerData']['federates'])
		"""
		# broker modifications
		if 'run_broker' in userConfig and not userConfig['run_broker']:
			userConfig['externally_connected_federates'].append('broker')

		# modifications for use_oedisi_runtime
		if not userConfig['use_oedisi_runtime']:
			userConfig['externally_connected_federates'].extend(self.config['oedisi_runtime_federates'])
			userConfig['externally_connected_federates'].append('sensor_voltage_magnitude')

		# config_runner modifications for externally_connected_federates
		componentsIndToBeRemoved=[]
		for thisExternallyConnectedFederate in userConfig['externally_connected_federates']:
			for n in range(len(config_runner['federates'])):
				if config_runner['federates'][n]['name']==thisExternallyConnectedFederate:
					componentsIndToBeRemoved.append(n)
					break
		componentsIndToBeRemoved.sort(reverse=True)
		print(f"Following components will be removed:{componentsIndToBeRemoved}")
		for thisInd in componentsIndToBeRemoved:
			print(f"Removing federates:{thisInd}")
			config_runner['federates'].pop(thisInd)

		# config_runner modifications for state estimator
		#if set(appFederates).difference(['state_estimator_nrel']):
		#	for appWithUpdate in set(appFederates).difference(['state_estimator_nrel']):
		#		for n in range(len(config_runner['federates'])):
		#			if config_runner['federates'][n]['name']==appWithUpdate:
		#				print(f"Updating application:{appWithUpdate}")
		#				config_runner['federates'][n]=availableFederates[appWithUpdate]
		#				break

		# modifications for user defined dsse and dopf federates
		userInterfaceSourceDir='/home/runtime/user_interface'
		for entry in userConfig['user_provided_federates']:
			userInterfaceDestinationDir=f'/home/run/{entry["name"]}'

			# copy
			directive=f'cp {os.path.join(userInterfaceSourceDir,"user_federate.py")} {os.path.join(userInterfaceSourceDir,"iohelper.py")} '+\
				f'{os.path.join(userInterfaceSourceDir,"exceptionutil.py")} {os.path.join(userInterfaceSourceDir,"config_user_federate.json")} '+\
				f'{entry["filepath"]} {userInterfaceDestinationDir}'
			os.system(directive)

			# update
			directive=f'mv {os.path.join(userInterfaceDestinationDir,"config_user_federate.json")} {os.path.join(userInterfaceDestinationDir,"config.json")} && '+\
				f'sed -i s/placeholderName/{entry["name"]}/g {os.path.join(userInterfaceDestinationDir,"config.json")} && '+\
				f'sed -i s/placeholderApplicationType/{entry["application_type"]}/g {os.path.join(userInterfaceDestinationDir,"config.json")}'
			os.system(directive)

		# config_runner modifications for user federate 
		for thisFederate in userConfig['user_provided_federates']:
			for n in range(len(config_runner['federates'])):
				if config_runner['federates'][n]['name']==thisFederate['name']:
					config_runner['federates'][n]['directory']=thisFederate['name']
					config_runner['federates'][n]['exec']='python user_federate.py'

		# modifications to config_runner for datapreprocessor federates
		if self.config['userConfig']['use_oedisi_preprocessor']:
			for datapreprocessorFederate in self.config['userConfig']['oedisi_preprocessor_federates']:
				if datapreprocessorFederate.replace("_", "").lower() =='dataimputation':
					print("Adding dataimputation federate...")
					config_runner['federates'].append({"directory": "dataimputation","name": "dataimputation","exec": "python federate_dataimputation.py","hostname": "localhost"})				
				elif datapreprocessorFederate.replace("_", "").lower() =='anomalydetection':
					print("Adding anomalydetection federate...")
					config_runner['federates'].append({"directory": "anomalydetection","name": "anomalydetection","exec": "python federate_anomalydetection.py","hostname": "localhost"})
				else:
					print(f"Datapreprocessor federate:{datapreprocessorFederate} is not supported!")

			for n in range(len(config_runner['federates'])):
				if config_runner['federates'][n]['name']=='broker':
					config_runner['federates'][n]['exec']=\
						f'helics_broker -f {len(config_runner["federates"])-1} --loglevel=warning'

		# write config_runner
		json.dump(config_runner,open('/home/run/system_runner.json','w'),indent=3) #Save modified syster_runner.json

		"""
		# patch for PNNL state estimator #not needed for now
		if 'state_estimator_pnnl' in userConfig['federates']:
			for entry in stateEstimatorWiringData['state_estimator_pnnl']['components']:
				print(f"Entry:{entry}")
				if entry['name']!='state_estimator_pnnl':
					print(f"Entry:{entry['name']}")
					if entry['type'].lower()=='recorder':
						thisPath=os.path.join('/home/run',entry['name'],'input_mapping.json')
						thisInputMapping=json.load(open(thisPath))
						if 'voltage_magnitude' in entry['name']:
							thisInputMapping={"subscription": "pnnl_state_estimator/Vmag_SE"}
						elif 'voltage_angle' in entry['name']:
							thisInputMapping={"subscription": "pnnl_state_estimator/Vang_SE"}
						json.dump(thisInputMapping,open(thisPath,'w'))
		"""

	def generateloadprofiles(self): #Method to generate load profiles for nodes in distribution system
		if self.config['userConfig']['use_oedisi_preprocessor']:
			preprocessorFederates=build.config['userConfig']['oedisi_preprocessor_federates']
			if any(federate in build.config['userConfig']['oedisi_preprocessor_federates'] for federate in ['nodeload', 'loadshape', 'loadprofile', 'load_profile']):
				basePath='/home/datapreprocessor/datapreprocessor/app/nodeload'
				filePath=f'{basePath}/generate_solar_node_load_profile_from_solarhome_data.py'
				loadProfilePath='/home/oedisi/oedisi-ieee123/profiles/load_profiles'
				opendssLocation=self.config['userConfig']['simulation_config']['opendss_location']+'/master.dss' ####TODO master.dss

				os.system(f'rm {basePath}/*.csv')
				directive=f'python {filePath} -d {opendssLocation} -n -1 -p {loadProfilePath} --fill 35040'
				print(f"Executing following directive in for generating load profiles:{directive}")
				flag=os.system(f'{directive}')
				assert flag==0,f'nodeload directive failed with flag={flag}'



if __name__=="__main__":
	parser=argparse.ArgumentParser()
	parser.add_argument('-c','--config',help='config to be passed to runner',required=False,
		default=None)
	args=parser.parse_args()

	build=Build(userConfigPath=args.config)
	build.write_config()
	build.generateloadprofiles()


