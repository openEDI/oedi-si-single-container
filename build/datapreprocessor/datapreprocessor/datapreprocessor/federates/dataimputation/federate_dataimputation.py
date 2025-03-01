import os
import json

import helics as h

from datapreprocessor.federates.dataimputation.iohelper import IOHelper
from datapreprocessor.app.dataimputation.data_imputation_postprocessing import update_window_and_impute
from datapreprocessor.app.model_utilities.model_save_load_utilities import modelarchive_to_modelpath,load_keras_model,load_tf_savedmodel
from datapreprocessor.utils.exceptionutil import ExceptionUtil
import datapreprocessor
workDir=datapreprocessor.__path__[0]

LogUtil=ExceptionUtil()
logFilePath=os.path.join(LogUtil.logDir,f'federate_dataimputation_process.log')
LogUtil.create_logger('federate_dataimputation_logger',logFilePath=logFilePath,logLevel=20,mode='w')

class DataimputationFederate(IOHelper):

	def __init__(self,config,federate_name='federate_dataimputation',dt=1):
		try:
			self.config=config
			for thisConf in ['component_definition','static_inputs','input_mapping']:
				self.config.update({thisConf:json.load(open(os.path.join(workDir,'federates','dataimputation',f'{thisConf}.json')))})
			self.config['federate_config']['subscriptions']=self.config['input_mapping']['subscriptions']
			self.federate_name = federate_name
			self.dt=dt
			model_folder = os.path.join(workDir,"app","dataimputation","model")
			model_archivepath = os.path.join(workDir,self.config['static_inputs']['model_archivepath'])
			self.model_path = modelarchive_to_modelpath(model_archivepath,model_folder)
			self.model_format = self.config['static_inputs']['model_format']
			LogUtil.logger.info('created dataimputation federate')
			LogUtil.logger.info(f"config::::{self.config}")
		except:
			LogUtil.exception_handler(raiseException=False)

#===================================================================================================
	def initialize(self):
		try:
			self.config['inputPort2TypeMapping']={}
			if 'dynamic_inputs' in self.config['component_definition']:
				for thisInput in self.config['component_definition']['dynamic_inputs']:
					self.config['inputPort2TypeMapping'][thisInput['port_id']]=thisInput['type']

			self.config['outputPort2TypeMapping']={}
			if 'dynamic_outputs' in self.config['component_definition']:
				for thisOutput in self.config['component_definition']['dynamic_outputs']:
					self.config['outputPort2TypeMapping'][thisOutput['port_id']]=thisOutput['type']
			
			self.create_federate(json.dumps(self.config['federate_config']))
			self.setup_publications(self.config['federate_config'])
			self.setup_subscriptions(self.config['federate_config'])
			print(f"Loading data imputation model from {self.model_path}")
			if self.model_format == "keras":
				self.autoencoder_dict =  {'pdemand':load_keras_model(self.model_path),'qdemand':load_keras_model(self.model_path)}
				self.autoencoder_dict['pdemand'].summary(expand_nested=True) #Summary only works for keras models
			elif self.model_format == "tfsm":
				self.autoencoder_dict =  {'pdemand':load_tf_savedmodel(self.model_path),'qdemand':load_tf_savedmodel(self.model_path)}
			else:
				raise ValueError(f"{self.model_format} is not a valid model format!")
			
			self.window_size = self.config['static_inputs']['window_size']
			self.input_features = self.config['static_inputs']['input_features']
			
			self.node_data_dict = {"pdemand":{node_id:{"data_raw_window":[self.config['static_inputs']['initial_measurements'][node_id]]*self.window_size,\
				"data_ffill_window":[self.config['static_inputs']['initial_measurements'][node_id]]*self.window_size,\
				"hour_window":[0]*self.window_size,"timestamp_window":[0.0]*self.window_size} for node_id in self.config['static_inputs']['monitored_nodes']},\
				"qdemand":{node_id:{"data_raw_window":[self.config['static_inputs']['initial_measurements'][node_id]]*self.window_size,\
				"data_ffill_window":[self.config['static_inputs']['initial_measurements'][node_id]]*self.window_size,\
				"hour_window":[0]*self.window_size,"timestamp_window":[0.0]*self.window_size} for node_id in self.config['static_inputs']['monitored_nodes']}}
			self.streaming_data_dict = {node_id:{"pdemand":{"data_ffill":0.0},"qdemand":{"data_ffill":0.0}} for node_id in self.config['static_inputs']['monitored_nodes']}
			self.window_id = 0

			LogUtil.logger.info('completed init')
		except:
			LogUtil.exception_handler()

#===================================================================================================
	def simulate(self,simEndTime=None):
		try:
			if not simEndTime:
				simEndTime=self.config['simulation_config']['end_time']
			self.enter_execution_mode()
			LogUtil.logger.info('entered execution mode')

			grantedTime=0
			grantedTime = h.helicsFederateRequestTime(self.federate,grantedTime)

			while grantedTime<simEndTime:
				# subscriptions
				subData=self.get_subscriptions(sub=self.sub,config=self.config,returnType='dict')
				LogUtil.logger.info(f"Completed subscription,{subData}")
				
				if subData:
					powers_real=subData['powers_real'].dict()
					powers_imag=subData['powers_imag'].dict()

					pdemand={'s'+k.replace('.1','a').replace('.2','b').replace('.3','c'):v \
						for k,v in zip(powers_real['ids'],powers_real['values'])}
					qdemand={'s'+k.replace('.1','a').replace('.2','b').replace('.3','c'):v \
						for k,v in zip(powers_imag['ids'],powers_imag['values'])}
					thisTimeStamp=powers_real['time']
					LogUtil.logger.debug(f"timestamp::::{thisTimeStamp}")
					commData={'pdemand':pdemand,'qdemand':qdemand}

					# run alg					
					preprocessed_output_dict = {node_id:{"pdemand":{},"qdemand":{}} for node_id in self.config['static_inputs']['monitored_nodes']} #create dictionary to hold output
					for monitored_node in self.config['static_inputs']['monitored_nodes']:
						LogUtil.logger.info(f"{monitored_node}:pdemand measured at {thisTimeStamp}:{commData['pdemand'][monitored_node]}")

						#Update pdemand ffill only if missing data is not detected
						if not commData['pdemand'][monitored_node] == 0.0: #Check if data is not missing
							self.streaming_data_dict[monitored_node]['pdemand'].update({"data_ffill":commData['pdemand'][monitored_node]})
						self.streaming_data_dict[monitored_node]['pdemand'].update({"timestamp":thisTimeStamp,"hour":thisTimeStamp.hour,"data_raw":commData['pdemand'][monitored_node]}) #update all other values

						#Update qdemand ffill only if missing data is not detected
						if not commData['qdemand'][monitored_node] == 0.0: #Check if data is not missing
							self.streaming_data_dict[monitored_node]['qdemand'].update({"data_ffill":commData['qdemand'][monitored_node]})
						self.streaming_data_dict[monitored_node]['qdemand'].update({"timestamp":thisTimeStamp,"hour":thisTimeStamp.hour,"data_raw":commData['qdemand'][monitored_node]}) #update all other values

						# pdemand -- Update output dict by calling imputation model
						preprocessed_output_dict[monitored_node]['pdemand'].update(update_window_and_impute(\
							self.streaming_data_dict[monitored_node]['pdemand'],self.autoencoder_dict['pdemand'],monitored_node,\
							self.window_size,self.node_data_dict['pdemand'],self.window_id ,self.input_features))

						# qdemand
						preprocessed_output_dict[monitored_node]['qdemand'].update(update_window_and_impute(\
							self.streaming_data_dict[monitored_node]['qdemand'],self.autoencoder_dict['qdemand'],\
							monitored_node,self.window_size,self.node_data_dict['qdemand'],self.window_id ,self.input_features))

					last_timestamp = list(preprocessed_output_dict[monitored_node]['pdemand'].keys())[-1]
					self.window_id = self.window_id + 1

					nodes= list(preprocessed_output_dict.keys())
					imputed_pdemand_measurements = {node:preprocessed_output_dict[node]['pdemand'][last_timestamp]['AE'] for node in nodes}
					imputed_qdemand_measurements = {node:preprocessed_output_dict[node]['qdemand'][last_timestamp]['AE'] for node in nodes}
					LogUtil.logger.info(f"Output from pdemand imputation model at {last_timestamp}:{imputed_pdemand_measurements}")
					LogUtil.logger.info(f"Output from qdemand imputation model at {last_timestamp}:{imputed_qdemand_measurements}")
					pubData={'powers_real':{'values':list(imputed_pdemand_measurements.values()),'ids':nodes,'equipment_ids':["di_1","di_2"],'time':last_timestamp},\
						     'powers_imag':{'values':list(imputed_qdemand_measurements.values()),'ids':nodes,'equipment_ids':["di_1","di_2"],'time':last_timestamp}}
					LogUtil.logger.info(f"Completed data imputation algorithm")

					# publications
					self.set_publications(pubData=pubData,pub=self.pub,config=self.config)
					LogUtil.logger.info(f"Completed data imputation publication")
				else:
					LogUtil.logger.info(f"No subscriptions found - data imputation not performed!")

				grantedTime = h.helicsFederateRequestTime(self.federate,grantedTime+1)
				LogUtil.logger.info(f'grantedTime::::{grantedTime}::::{subData.keys()}')
		except:
			LogUtil.exception_handler()


#===================================================================================================
if __name__=='__main__':
	try:
		config=json.load(open(os.path.join(workDir,'federates','dataimputation','config_dataimputation.json')))

		dif=DataimputationFederate(config)
		dif.initialize()
		dif.simulate()
		LogUtil.logger.info('completed simulation')
		dif.finalize()
	except:
		LogUtil.exception_handler(raiseException=True)


