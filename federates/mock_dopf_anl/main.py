import os
import json
import argparse
import logging
import pdb

import helics as h
import hs071

from iohelper import IOHelper


baseDir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
formatterStr='%(asctime)s::%(name)s::%(filename)s::%(funcName)s::'+\
	'%(levelname)s::%(message)s::%(threadName)s::%(process)d'
formatter = logging.Formatter(formatterStr)
fh = logging.FileHandler(os.path.join(baseDir,'logs','mock_dopf_anl.log'),mode='w')
fh.setFormatter(formatter)
fh.setLevel(10)
logger = logging.getLogger('mock_dopf_anl_logger')
logger.setLevel(10)
logger.addHandler(fh)


class MockApp(IOHelper):

	def __init__(self,config):
		self.config=config
		self.dt=self.config['period']

#=======================================================================================================================
	def initialize(self,config=None):
		if not config:
			config=self.config
		self.create_federate(json.dumps(config))
		self.setup_publications(config)
		self.setup_subscriptions(config)
		logger.info('completed init')

#=======================================================================================================================
	def simulate(self,simEndTime=24):
		self.enter_execution_mode()
		logger.info('entered execution mode')

		grantedTime=0
		while grantedTime<simEndTime:
			# sub
			subData={}
			for entry in self.sub:
				if h.helicsInputIsUpdated(self.sub[entry]['indexObj']):
					val=h.__dict__[self.sub[entry]['method']](self.sub[entry]['indexObj'])
					subData[entry]=val
					logger.info(f'grantedTime={grantedTime}::::subData={entry}:{val}')

			# run
			hs071.main()
			pubData={self.config['name']+'/'+entry['key']:100 for entry in self.config['publications']}
			logger.info(f'grantedTime={grantedTime}::::pubData={pubData}')

			# publish
			for entry in self.pub:
				h.__dict__[self.pub[entry]['method']](self.pub[entry]['indexObj'],pubData[entry])

			grantedTime=h.helicsFederateRequestTime(self.federate,grantedTime+self.dt)
			logger.info(f'grantedTime::::{grantedTime} with dt::::{self.dt}')

#=======================================================================================================================
if __name__=='__main__':
	parser=argparse.ArgumentParser()
	parser.add_argument('-c','--config',help='config file',default='config.json')
	parser.add_argument('-e','--end_time',default=1200)
	args=parser.parse_args()

	baseDir=os.path.dirname(os.path.abspath(__file__))
	config=json.load(open(os.path.join(baseDir,args.config)))

	thisFed=MockApp(config)
	thisFed.initialize()
	thisFed.simulate(simEndTime=args.end_time)
	thisFed.finalize()


