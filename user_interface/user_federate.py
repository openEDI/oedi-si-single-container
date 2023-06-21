import os
import json
import uuid
import copy
import datetime
import pdb

import helics as h
import gadal.gadal_types.data_types as GadalTypes

from iohelper import IOHelper
from exceptionutil import ExceptionUtil
from main import algorithm


LogUtil=ExceptionUtil()
baseDir=os.path.dirname(os.path.abspath(__file__))
thisUUID=uuid.uuid4().hex
logConfig={
	"logLevel": 20, 
	"logFilePath": os.path.join(baseDir,f'user_federate.log'), 
	"mode": "w", 
	"loggerName": "user_federate_logger"
}
LogUtil.create_logger(**logConfig)


class UserFederate(IOHelper):

	def __init__(self,config,dt=1):
		try:
			self.config=config
			for thisConf in ['component_definition','static_inputs','input_mapping']:
				self.config.update({thisConf:\
					json.load(open(os.path.join(baseDir,f'{thisConf}.json')))})

			# transform self.config['input_mapping']
			self.config['input_mapping_ext']=copy.copy(self.config['input_mapping'])
			input_mapping={"subscriptions":[]}
			for entry in self.config['input_mapping']:
				input_mapping['subscriptions'].append(\
					{'required':False,"key":self.config['input_mapping'][entry],"type":"string"})
			self.config['input_mapping']=input_mapping

			self.config['federate_config']['subscriptions']=self.config['input_mapping']['subscriptions']
			self.federate_name = self.config['federate_config']['name']
			self.dt=dt

			self.requiredOutput={'dsse':['VoltagesMagnitude','VoltagesAngle'],\
				'dopf':['VoltagesMagnitude','VoltagesAngle','PowersReal','PowersImaginary']}

			LogUtil.logger.info('created federate')
		except:
			LogUtil.exception_handler(raiseException=True)

#===================================================================================================
	def initialize(self):
		try:
			self.config['inputPort2TypeMapping']={}
			if 'dynamic_inputs' in self.config['component_definition']:
				for thisInput in self.config['component_definition']['dynamic_inputs']:
					self.config['inputPort2TypeMapping'][self.config['input_mapping_ext'][thisInput['port_id']]]=thisInput['type']

			self.config['outputPort2TypeMapping']={}
			if 'dynamic_outputs' in self.config['component_definition']:
				for thisOutput in self.config['component_definition']['dynamic_outputs']:
					self.config['outputPort2TypeMapping'][self.config['federate_config']['name']+'/'+thisOutput['port_id']]=thisOutput['type']

			self.create_federate(json.dumps(self.config['federate_config']))
			self.setup_publications(self.config['federate_config'])
			self.setup_subscriptions(self.config['federate_config'])

			self.config['pubKeyForTime']=[entry for entry in self.config['inputPort2TypeMapping'] if 'power' in entry or 'voltage' in entry]
			self.config['pubKeyForTime']=self.config['pubKeyForTime'][0]

			LogUtil.logger.info('completed init')
		except:
			LogUtil.exception_handler()

#===================================================================================================
	def simulate(self,algorithmObj,simEndTime=None):
		try:
			if not simEndTime:
				simEndTime=self.config['simulation_config']['end_time']
			self.enter_execution_mode()
			LogUtil.logger.info('entered execution mode')

			grantedTime=0
			grantedTime = h.helicsFederateRequestTime(self.federate,grantedTime)
			application_type=self.config['simulation_config']['application_type']

			while grantedTime<=simEndTime:
				# subscriptions
				subData=self.get_subscriptions(sub=self.sub,config=self.config)
				LogUtil.logger.info(f"Completed subscription with data {subData.keys()}")

				inputData={}
				for entry in subData:
					inputData[subData[entry].__class__.__name__]=subData[entry].dict()

				if subData:
					# run alg
					LogUtil.logger.info(f"inputData::::{inputData}")
					outputData=algorithmObj(inputData)
					assert not set(self.requiredOutput[application_type]).difference(outputData.keys()),\
						f"following required output is missing {set(self.requiredOutput[application_type]).difference(outputData.keys())}"
					LogUtil.logger.info(f"outputData::::{outputData.keys()},outputPort2TypeMapping::::{self.config['outputPort2TypeMapping']}")

					pubData={}
					for entry in self.config['outputPort2TypeMapping']:
						pubData[entry]=outputData[self.config['outputPort2TypeMapping'][entry]]
					LogUtil.logger.info(f"pubData::::{pubData}")

					# publications
					self.set_publications(pubData=pubData,pub=self.pub,config=self.config)
					LogUtil.logger.info("Completed publication")

				grantedTime = h.helicsFederateRequestTime(self.federate,grantedTime+1)

				if grantedTime<=simEndTime:
					LogUtil.logger.info(f'grantedTime::::{grantedTime}')
		except:
			LogUtil.exception_handler()


#===================================================================================================
if __name__=='__main__':
	try:
		config=json.load(open(os.path.join(baseDir,'config.json')))

		df=UserFederate(config)
		df.initialize()
		df.simulate(algorithmObj=algorithm)
		df.finalize()
	except:
		LogUtil.exception_handler(raiseException=True)


