import os
import json
import copy
import pdb
import argparse


class Build(object):

	def __init__(self,userConfigPath=None):
		self._baseDir=os.path.dirname(os.path.abspath(__file__))
		self.config={}
		self.config['availableFederates']=json.load(open(os.path.join(self._baseDir,'available_federates.json')))

		if not userConfigPath:
			userConfigPath=os.path.join(self._baseDir,'user_config.json')
		self.config['userConfig']=json.load(open(userConfigPath))

		self.config['links']=self._convertToOedisiLinks(json.load(open(os.path.join(self._baseDir,'links.json'))))
		self.config['oedisi_runtime_federates']=['feeder','sensor_voltage_real','sensor_voltage_imaginary',\
			'sensor_power_real','sensor_power_imaginary','recorder_voltage_real','recorder_voltage_imag']
		self.config['oedisi_runtime_wiring_diagram']=json.load(open(os.path.join(self._baseDir,'oedisi_runtime_wiring_diagram.json')))
		self.userFederateOptions={'allowedApplicationTypes':['dsse','dopf'],'supportedLanguages':['python']}



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

		# write wiring diagram and generate config_runner
		wiring_diagram_path=os.path.join(self._baseDir,'wiring_diagram.json')
		json.dump(wiringDiagramData,open(wiring_diagram_path,'w'),indent=3)
		directive=f'cd /home/oedisi/sgidal-example && python3 /home/oedisi/sgidal-example/test_full_systems.py '+\
			f'--system {wiring_diagram_path} --target-directory /home/run'
		flag=os.system(directive)
		assert flag==0,f'generating config_runner failed with flag:{flag}'

		config_runner=json.load(open('/home/run/test_system_runner.json'))

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

		# write config_runner
		json.dump(config_runner,open('/home/run/test_system_runner.json','w'),indent=3)

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


if __name__=="__main__":
	parser=argparse.ArgumentParser()
	parser.add_argument('-c','--config',help='config to be passed to runner',required=False,
		default=None)
	args=parser.parse_args()

	build=Build(userConfigPath=args.config)
	build.write_config()


