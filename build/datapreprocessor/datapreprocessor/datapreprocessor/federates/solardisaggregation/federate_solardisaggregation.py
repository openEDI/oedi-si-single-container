import os
import json

import helics as h

from datapreprocessor.federates.dataimputation.iohelper import IOHelper
from datapreprocessor.app.solardisaggregation.solardisaggregation_postprocessing import update_window_and_disaggregate
from datapreprocessor.app.model_utilities.model_save_load_utilities import modelarchive_to_modelpath,load_keras_model,load_tf_savedmodel
from datapreprocessor.utils.exceptionutil import ExceptionUtil
import datapreprocessor
workDir=datapreprocessor.__path__[0]

LogUtil=ExceptionUtil()
logFilePath=os.path.join(LogUtil.logDir,f'federate_solardisaggregation_process.log')
LogUtil.create_logger('federate_dataimputation_logger',logFilePath=logFilePath,logLevel=20,mode='w')

class SolarDisaggregationFederate(IOHelper):

	def __init__(self,config,federate_name='federate_solardisaggregation',dt=1):
		try:
			self.config=config
			for thisConf in ['component_definition','static_inputs','input_mapping']:
				self.config.update({thisConf:json.load(open(os.path.join(workDir,'federates','solardisaggregation',f'{thisConf}.json')))})
			print("Completed config update")
			self.config['federate_config']['subscriptions']=self.config['input_mapping']['subscriptions']
			self.federate_name = federate_name
			self.dt=dt
			model_folder = os.path.join(workDir,"app","solardisaggregation","model")
			model_archivepath = os.path.join(workDir,self.config['static_inputs']['model_archivepath'])
			self.model_path = modelarchive_to_modelpath(model_archivepath,model_folder)
			self.model_format = self.config['static_inputs']['model_format']			
			LogUtil.logger.info('created solar disaggregation federate')
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

			print(f"Loading solar dissagregation model from {self.model_path}")
			if self.model_format == "keras":
				self.predictor_dict =  {'pdemand':load_keras_model(self.model_path)}
				self.predictor_dict['pdemand'].summary(expand_nested=True) #Summary only works for keras models
			elif self.model_format == "tfsm":
				self.predictor_dict =  {'pdemand':load_tf_savedmodel(self.model_path)}
			else:
				raise ValueError(f"{self.model_format} is not a valid model format!")			
			
			self.window_size = self.config['static_inputs']['window_size']
			self.input_features = self.config['static_inputs']['input_features']
			
			self.node_data_dict = {"pdemand":{node_id:{"data_raw_window":[self.config['static_inputs']['initial_measurements'][node_id]]*self.window_size,
													   "hour_window":[0]*self.window_size,
													   "day_of_week_window":[0]*self.window_size,
													   "timestamp_window":[0.0]*self.window_size} for node_id in self.config['static_inputs']['monitored_solar_nodes']}								   
								  }
			self.streaming_data_dict = {node_id:{"pdemand":{"data_raw":0.0}} for node_id in self.config['static_inputs']['monitored_solar_nodes']}
			self.window_id = 0
			
			LogUtil.logger.info('completed init for solar disaggregation')
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
				LogUtil.logger.info(f"Completed subscription")
				
				if subData:
					powers_real=subData['powers_real'].dict()					
					
					pdemand={'s'+k.replace('.1','a').replace('.2','b').replace('.3','c'):v for k,v in zip(powers_real['ids'],powers_real['values'])}					
					thisTimeStamp=powers_real['time']
					LogUtil.logger.debug(f"timestamp::::{thisTimeStamp}")
					commData={'pdemand':pdemand}
									
					# run alg
					disaggregated_output_dict = {node_id:{"pdemand":{thisTimeStamp:{}}} for node_id in self.config['static_inputs']['monitored_solar_nodes']} #create dictionary to hold output
					
					for monitored_node in self.config['static_inputs']['monitored_solar_nodes']:						
						self.streaming_data_dict[monitored_node]['pdemand'].update({"timestamp":thisTimeStamp,"data_raw":commData['pdemand'][monitored_node],"hour":thisTimeStamp.hour,"day_of_week":thisTimeStamp.weekday()}) #update all other value						  
						disaggregated_output_dict[monitored_node]['pdemand'].update(update_window_and_disaggregate(self.streaming_data_dict[monitored_node]['pdemand'],self.predictor_dict['pdemand'],monitored_node,
																												  self.window_size,self.node_data_dict['pdemand'],self.window_id,self.input_features)) #Update output dict by calling disaggregation model
					
					last_timestamp = list(disaggregated_output_dict[monitored_node]['pdemand'].keys())[-1]
					self.window_id = self.window_id + 1
					#print(f"Output dict:{disaggregated_output_dict}")
					#print(f"Last time stamp:{last_timestamp}")

					nodes= list(disaggregated_output_dict.keys())
					disaggregated_solar_measurements = {node:disaggregated_output_dict[node]['pdemand'][last_timestamp]['prediction'] for node in nodes}
					#print(f"Nodes:{nodes}")
					LogUtil.logger.info(f"Output from pdemand solar disaggregation model at {last_timestamp}:{disaggregated_solar_measurements}")
					pubData={'powers_real':{'values':list(disaggregated_solar_measurements.values()),'ids':nodes,'equipment_ids':["di_1","di_2"],'time':last_timestamp}}
					LogUtil.logger.info(f"Completed solar disaggregation algorithm")

					# publications
					self.set_publications(pubData=pubData,pub=self.pub,config=self.config)
					LogUtil.logger.info(f"Completed solar disaggregation publication")
				else:
					LogUtil.logger.info(f"No subscriptions found - solar disaggregation not performed!")

				grantedTime = h.helicsFederateRequestTime(self.federate,grantedTime+1)
				LogUtil.logger.info(f'grantedTime::::{grantedTime}::::{subData.keys()}')
		except:
			LogUtil.exception_handler()


#===================================================================================================
if __name__=='__main__':
	try:
		config=json.load(open(os.path.join(workDir,'federates','solardisaggregation','config_solardisaggregation.json')))

		dif=SolarDisaggregationFederate(config)
		dif.initialize()
		dif.simulate()
		LogUtil.logger.info('completed solar dissagregation simulation')
		dif.finalize()
	except:
		LogUtil.exception_handler(raiseException=True)


