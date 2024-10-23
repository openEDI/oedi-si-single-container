import os
import uuid
import json
import sys
from collections import OrderedDict, defaultdict

import helics as h
import oedisi.types.data_types as OedisiTypes

from datapreprocessor.utils.exceptionutil import ExceptionUtil

LogUtil=ExceptionUtil()
logFilePath=os.path.join(LogUtil.logDir,f'iohelper_{uuid.uuid4().hex}.log')
LogUtil.create_logger('iohelper_logger',logFilePath=logFilePath,logLevel=20,mode='w')


class IOHelper(object):
	start_time: int = -1
	end_time: int = -1
	step_time: int = 1

	federate_info_core_name: str = ""
	federate_info_core_init_string: str = "--federates=1"
	federate_info_core_type: str = "zmq"

	federate_info_time_delta: float = 1.00
	federate_info_logging_level: int = 1

	federate_name: str = ""

	def __init__(self):
		self.federate = None
		self.federate_info = None
		self.data = {}

		if self.federate_info_core_name == "":
			self.federate_info_core_name = self.federate_name

	def create(self):
		self.create_federate()
		self.register_interfaces()

	def initialize(self):
		pass

	def create_federate(self,config):
		self.federate=h.helicsCreateValueFederateFromConfig(config)

	def register_interfaces(self):
		self.register_publications()
		self.register_subscriptions()
		self.register_endpoints()

#===================================================================================================
	def register_publications(self,**kwargs):
		for thisPub in kwargs:
			self.federate.register_publication(**kwargs[thisPub])

#===================================================================================================
	def register_subscriptions(self,**kwargs):
		for thisSub in kwargs:
			self.federate.register_subscription(**kwargs[thisSub])

	def register_endpoints(self):
		pass

	def version(self):
		h.helicsGetVersion()

	def next_time(self, t):
		res=self.federate.request_time_advance(t)
		return res

#===================================================================================================
	def setup_publications(self,config):
		self.pub=pub={}
		if 'publications' in config:
			LogUtil.logger.info("publication list::::{}".format([thisSub['key'] for thisSub in config['publications']]))
			for index,item in zip(range(h.helicsFederateGetPublicationCount(self.federate)),config['publications']):
				indexObj=h.helicsFederateGetPublicationByIndex(self.federate,index)
				thisPub=pub[h.helicsPublicationGetKey(indexObj)]={}
				thisPub['indexObj']=indexObj
				if item['type']=='bool':
					thisPub['method']='helicsPublicationPublishBoolean'
				elif item['type']=='double':
					thisPub['method']='helicsPublicationPublishDouble'
				elif item['type']=='string':
					thisPub['method']='helicsPublicationPublishString'
		LogUtil.logger.info("added publications")

#===================================================================================================
	def setup_subscriptions(self,config):
		self.sub=sub={}
		if 'subscriptions' in config:
			LogUtil.logger.info("subscription list::::{}".format([thisSub['key'] for thisSub in config['subscriptions']]))
			for index,item in zip(range(h.helicsFederateGetInputCount(self.federate)),config['subscriptions']):
				indexObj=h.helicsFederateGetInputByIndex(self.federate,index)
				thisSub=sub[h.helicsInputGetKey(indexObj)]={}
				thisSub['indexObj']=indexObj
				thisSub['key']=item['key']
				if item['type']=='bool':
					thisSub['method']='helicsInputGetBoolean'
				elif item['type']=='double':
					thisSub['method']='helicsInputGetDouble'
				elif item['type']=='string':
					thisSub['method']='helicsInputGetString'
		LogUtil.logger.info("added subscriptions")

#===================================================================================================
	def get_subscriptions(self,sub,config,**kwargs):
		subData={}
		for entry in sub:
			if h.helicsInputIsUpdated(sub[entry]['indexObj']):
				thisKey=sub[entry]['indexObj'].target.split('/')[-1]
				val=h.__dict__[sub[entry]['method']](sub[entry]['indexObj'])
				if thisKey not in subData:
					subData[thisKey]={}
				assert thisKey in config['inputPort2TypeMapping'],\
					f'{thisKey} not in inputPort2TypeMapping'
				assert config['inputPort2TypeMapping'][thisKey] in \
					OedisiTypes.__dict__,f'{config["inputPort2TypeMapping"][thisKey]} not in OedisiTypes'
				subData[thisKey] = OedisiTypes.__dict__[\
					config['inputPort2TypeMapping'][thisKey]].parse_obj(json.loads(val))
		return subData


#===================================================================================================
	def set_publications(self,pubData,pub,config,**kwargs):
		for entry in pub:
			thisKey=entry.split('/')[-1]
			assert isinstance(pubData[thisKey],dict),f'data for {thisKey} is not a dictionary'
			assert thisKey in config['outputPort2TypeMapping'],f'{thisKey} not in outputPort2TypeMapping'
			assert config['outputPort2TypeMapping'][thisKey] in OedisiTypes.__dict__,\
				f'{config["outputPort2TypeMapping"][thisKey]} not in OedisiTypes'
			thisData=OedisiTypes.__dict__[\
			config['outputPort2TypeMapping'][thisKey]](**pubData[thisKey]).json()
			h.__dict__[pub[entry]['method']](pub[entry]['indexObj'],thisData)

#===================================================================================================
	def publications(self,**kwargs):
		success=True
		for thisPub in kwargs:
			if thisPub in self.federate.publications:
				self.federate.publications[thisPub].publish(kwargs[thisPub])
			else:
				success=False
				break
		LogUtil.logger.info("added subscriptions")
		return success


	def enter_execution_mode(self):
		self.federate.enter_executing_mode()

#===================================================================================================
	def finalize(self):
		try:
			h.helicsFederateFree(self.federate)
			h.helicsCloseLibrary()
			LogUtil.logger.info('completed finalizing helics')
		except:
			LogUtil.exception_handler(raiseException=False)

#=======================================================================================================================
	def simulate(self):
		self.initialize()
		self.create()
		self.setup()
		self.run()

#===================================================================================================
	def start_broker(self,nFeds):
		try:
			initstring = "-f {} --name=mainbroker".format(nFeds)
			self.broker = h.helicsCreateBroker("zmq", "", initstring)
			assert h.helicsBrokerIsConnected(self.broker)==1,"broker connection failed"
		except:
			LogUtil.exception_handler(raiseException=False)



