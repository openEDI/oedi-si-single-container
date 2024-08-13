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
		unavailableFed=set(userConfig['federates']).difference(availableFederates)
		assert not unavailableFed,f"federates {unavailableFed} are unavailable"

		# generate wiring diagram
		appFederates=list(set(userConfig['federates']).difference(self.config['oedisi_runtime_federates']))

		wiringDiagramData={'name':userConfig['name'],'components':[],'links':[]}
		# add components even if userConfig['use_oedisi_runtime'] is False and remove later as needed
		temp=copy.deepcopy(self.config['oedisi_runtime_wiring_diagram'])

		for n in range(len(temp['components'])):
			if temp['components'][n]['name']=='feeder':
				temp['components'][n]['parameters'].update(simulationConfig)

		wiringDiagramData['components'].extend(temp['components'])
		wiringDiagramData['links'].extend(temp['links'])

		for thisAppFederate in appFederates:
			assert thisAppFederate in stateEstimatorWiringData,f'{thisAppFederate} not in stateEstimatorWiringData'
			wiringDiagramData['components'].extend(stateEstimatorWiringData[thisAppFederate]['components'])
			wiringDiagramData['links'].extend(stateEstimatorWiringData[thisAppFederate]['links'])

		# make mods for preprocessor
		if self.config['userConfig']['use_oedisi_preprocessor']:
			preprocessorFederates=build.config['userConfig']['oedisi_preprocessor_federates']
			preprocessorFederatesDir='/home/datapreprocessor/datapreprocessor/federates'
			availablePreprocessorFederates=os.listdir(preprocessorFederatesDir)
			if not os.path.exists('/home/run'):
				os.system('mkdir /home/run')
			for thisFederate in preprocessorFederates:
				thisFederate=thisFederate.replace('-','').replace('_','') #### TODO: foo_bar to foobar
				if thisFederate in availablePreprocessorFederates:
					flag=os.system(f'cp -r {os.path.join(preprocessorFederatesDir,thisFederate)} /home/run')
					assert flag==0, f'copying {thisFederate} failed with error flag={flag}'

		# template based federate additions to wiring diagram (dopf_ornl)
		#### TODO: Need to make changes to config data structure to include dopf federates
		thisConf=self._get_template_federate_config('dopf_ornl','StateEstimatorComponent',\
			'/home/run/dopf_ornl/','python /home/run/dopf_ornl/dopf_federate.py',src='/home/dopf_ornl/')
		self._update_wiring_diagram_data(wiringDiagramData,thisConf)

		# write wiring diagram and generate config_runner
		wiring_diagram_path=os.path.join(self._baseDir,'wiring_diagram.json')
		json.dump(wiringDiagramData,open(wiring_diagram_path,'w'),indent=3)
		#directive=f'cd /home/oedisi/oedi-example && python3 /home/oedisi/oedi-example/test_full_systems.py '+\
		#	f'--system {wiring_diagram_path} --target-directory /home/run'
		directive=f'oedisi build --target-directory /home/run --component-dict /home/oedisi/oedisi-example/components.json --system /home/oedisi/oedisi-example/scenarios/docker_system.json'
		flag=os.system(directive)
		assert flag==0,f'generating config_runner failed with flag:{flag}'

		# template based federate additions (dopf_ornl)
		if thisConf['data']['src']:# replace template data after wiring diagram puts default data
			os.system(f'rm -r /home/run/{thisConf["configRunnerData"]["federates"]["name"]}/* && '+\
				f'cp -r {thisConf["data"]["src"]}/* /home/run/{thisConf["configRunnerData"]["federates"]["name"]}')

		config_runner=json.load(open('/home/run/system_runner.json'))

		# config_runner mods for dopf_ornl
		ind=-1
		for n in range(len(config_runner['federates'])):
			if config_runner['federates'][n]['name']==thisConf['configRunnerData']['federates']['name']:
				ind=n
		if ind>=0:
			config_runner['federates'].pop(ind)
			config_runner['federates'].append(thisConf['configRunnerData']['federates'])

		# broker mods
		if 'run_broker' in userConfig and not userConfig['run_broker']:
			userConfig['externally_connected_federates'].append('broker')

		# mods for use_oedisi_runtime
		if not userConfig['use_oedisi_runtime']:
			userConfig['externally_connected_federates'].extend(self.config['oedisi_runtime_federates'])
			userConfig['externally_connected_federates'].append('sensor_voltage_magnitude')

		# mods for externally_connected_federates
		componentsIndToBeRemoved=[]
		for thisExternallyConnectedFederate in userConfig['externally_connected_federates']:
			for n in range(len(config_runner['federates'])):
				if config_runner['federates'][n]['name']==thisExternallyConnectedFederate:
					componentsIndToBeRemoved.append(n)
					break
		componentsIndToBeRemoved.sort(reverse=True)
		for thisInd in componentsIndToBeRemoved:
			config_runner['federates'].pop(thisInd)

		# mods for state estimator
		if set(appFederates).difference(['state_estimator_nrel']):
			for appWithUpdate in set(appFederates).difference(['state_estimator_nrel']):
				for n in range(len(config_runner['federates'])):
					if config_runner['federates'][n]['name']==appWithUpdate:
						config_runner['federates'][n]=availableFederates[appWithUpdate]
						break

		# mods for user defined dsse and dopf federates
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

		# user federate mods
		for thisFederate in userConfig['user_provided_federates']:
			for n in range(len(config_runner['federates'])):
				if config_runner['federates'][n]['name']==thisFederate['name']:
					config_runner['federates'][n]['directory']=thisFederate['name']
					config_runner['federates'][n]['exec']='python user_federate.py'

		# mods for preprocessor
		if self.config['userConfig']['use_oedisi_preprocessor']:
			if 'data_imputation' in self.config['userConfig']['oedisi_preprocessor_federates']:
				config_runner['federates'].append({"directory": "dataimputation","name": "dataimputation",\
					"exec": "python federate_dataimputation.py","hostname": "localhost"})

			for n in range(len(config_runner['federates'])):
				if config_runner['federates'][n]['name']=='broker':
					config_runner['federates'][n]['exec']=\
						f'helics_broker -f {len(config_runner["federates"])-1} --loglevel=warning'

		# write config_runner
		json.dump(config_runner,open('/home/run/system_runner.json','w'),indent=3)

		# patch for PNNL state estimator
		if 'state_estimator_pnnl' in userConfig['federates']:
			for entry in stateEstimatorWiringData['state_estimator_pnnl']['components']:
				if entry['name']!='state_estimator_pnnl':
					if entry['type'].lower()=='recorder':
						thisPath=os.path.join('/home/run',entry['name'],'input_mapping.json')
						thisInputMapping=json.load(open(thisPath))
						if 'voltage_magnitude' in entry['name']:
							thisInputMapping={"subscription": "pnnl_state_estimator/Vmag_SE"}
						elif 'voltage_angle' in entry['name']:
							thisInputMapping={"subscription": "pnnl_state_estimator/Vang_SE"}
						json.dump(thisInputMapping,open(thisPath,'w'))


	def preprocessor(self):
		if self.config['userConfig']['use_oedisi_preprocessor']:
			preprocessorFederates=build.config['userConfig']['oedisi_preprocessor_federates']
			if 'nodeload' in preprocessorFederates or 'loadshape' in preprocessorFederates or \
				'loadprofile' in preprocessorFederates or 'load_profile' in preprocessorFederates:
				basePath='/home/datapreprocessor/datapreprocessor/app/nodeload'
				filePath=f'{basePath}/generate_solar_node_load_profile_from_solarhome_data.py'
				loadProfilePath='/home/oedisi/oedisi-ieee123/profiles/load_profiles'
				opendssLocation=self.config['userConfig']['simulation_config']['opendss_location']+'/master.dss' ####TODO master.dss

				os.system(f'rm {basePath}/*.csv')
				directive=f'python {filePath} -d {opendssLocation} -n -1 -p {loadProfilePath} --fill 35040'
				flag=os.system(f'{directive}')
				assert flag==0,f'nodeload directive failed with flag={flag}'



if __name__=="__main__":
	parser=argparse.ArgumentParser()
	parser.add_argument('-c','--config',help='config to be passed to runner',required=False,
		default=None)
	args=parser.parse_args()

	build=Build(userConfigPath=args.config)
	build.write_config()
	build.preprocessor()


