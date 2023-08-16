import os
import math
import copy
import json
import pdb
import socket
import datetime
import logging
import uuid

import dss as odss
import numpy as np
import networkx as nx
import pandas as pd

from datapreprocessor.utils.exceptionutil import ExceptionUtil


LogUtil=ExceptionUtil()
baseDir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logConfig={
	"logLevel": 20, 
	"logFilePath": os.path.join(baseDir,'logs','distmodel.log'), 
	"mode": "w", 
	"loggerName": "oedi_distmodel_logger"
}
LogUtil.create_logger(**logConfig)


#=======================================================================================================================
#=======================================================================================================================
#=======================================================================================================================
class DistModel(object):

	def __init__(self,serverMode=False):
		try:
			self.startingDir=os.getcwd()
			self._engine=odss.DSS
			self.Circuit=self._engine.ActiveCircuit
			self.Text = self._engine.Text
			self.Solution = self.Circuit.Solution
			self.CktElement = self.Circuit.ActiveCktElement
			self.Bus = self.Circuit.ActiveBus
			self.Meters = self.Circuit.Meters
			self.PDElement = self.Circuit.PDElements
			self.Loads = self.Circuit.Loads
			self.Lines = self.Circuit.Lines
			self.Transformers = self.Circuit.Transformers
			self.Monitors = self.Circuit.Monitors

			self.dermap={}
			self.busname2ind={}

			self.baseDir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

			self.casefilePath={}
			self.casefilePath['case13']=os.path.join(self.baseDir,'data','opendss','13Bus','case13ZIP.dss')
			self.casefilePath['case123']=os.path.join(self.baseDir,'data','opendss','123Bus','case123ZIP.dss')
			self.casefilePath['case8500']=os.path.join(self.baseDir,'data','opendss','8500Bus','master.dss')
			self.casefilePath['casesmartdssmall']=os.path.join(self.baseDir,'data','opendss','smartds_small','opendss','Master.dss')
			self.casefilePath['casesmartdsmedium']=os.path.join(self.baseDir,'data','opendss','smartds_medium','opendss','Master.dss')
			self.casefilePath['casesmartdslarge']=os.path.join(self.baseDir,'data','opendss','smartds_large','opendss','Master.dss')

			self.loadshape=json.load(open(os.path.join(self.baseDir,'data','loadshape','caiso_2015_1hr.json')))

			self.OPTIMIZE_CONTROL_EFFORT=1
			self.OPTIMIZE_UNBALANCE=2

			self.assetData={}

			if serverMode:
				self.server_start()
		except:
			LogUtil.exception_handler()

#=======================================================================================================================
	def server_start(self,hostIP='127.0.0.1',portNum=11050):
		try:
			self.s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
			self.s.bind((hostIP,portNum))
			self.s.listen(0)
			self.c=self.s.accept()[0] # expects only a single connection
		except:
			LogUtil.exception_handler()

#=======================================================================================================================
	def server_send(self,msg,addEOF=True):
		try:
			msgStr=json.dumps(msg)
			if addEOF:
				msgStr=msgStr+'#'
			self.c.send(msgStr.encode())
		except:
			LogUtil.exception_handler()

#=======================================================================================================================
	def server_recv(self,bufferSize=1024,isJson=True):
		try:
			msg=b' '
			while msg[-1]!=ord('#'):
				msg+=self.c.recv(bufferSize)
			if isJson:
				msg=json.loads(msg[0:-1].decode())

			return msg
		except:
			LogUtil.exception_handler()

#=======================================================================================================================
	def server_close(self):
		try:
			self.s.close()
		except:
			LogUtil.exception_handler()

#=======================================================================================================================
	def msg2procedure(self):
		try:
			# recv msg
			msg=self.server_recv()

			# process
			success=False; output=''; comm_end=False;
			if 'method' in msg and msg['method'] in DistModel.__dict__:
				output=DistModel.__dict__[msg['method']](self,**msg['args'])
				success=True
			elif 'comm_end' in msg and msg['comm_end']:
				success=True
				comm_end=True

			# reply
			reply={'success':success,'output':output}
			self.server_send(reply)

			return comm_end
		except:
			reply={'success':False,'output':'','serverError':True}
			self.server_send(reply)
			LogUtil.exception_handler(raiseException=False)

#=======================================================================================================================
	def load_case(self,fpath):
		try:
			# Always a good idea to clear the DSS before loading a new circuit
			self._engine.ClearAll()
			self.assetData={}

			if not os.path.exists(fpath) and fpath in self.casefilePath:
				fpath=self.casefilePath[fpath]
			self.currentFpath=fpath

			self.Text.Command = f'Redirect [{fpath}]'
			LogUtil.logger.info(f'Executed text command "Redirect [{fpath}]"')
			os.chdir(self.startingDir)# openDSS sets dir to data dir, revert back

			# load assetData if available
			if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(fpath)),'assetData.json')):
				self.assetData=json.load(open(os.path.join(os.path.dirname(os.path.abspath(fpath)),'assetData.json')))

			# once loaded find the base load
			self.S0=self._getLoads()

			# get info
			for n in range(0,self.Circuit.NumBuses):
				self.busname2ind[self.Circuit.Buses(n).Name]=n

			# get info
			self.Loads.First # start with the first load in the list
			self.ind2loadname={}
			self.loadname2ind={}
			for n in range(self.Loads.Count):
				# will use 1-index as self.Loads.idx uses 1-index
				self.ind2loadname[n+1]=self.Loads.Name
				self.loadname2ind[self.Loads.Name]=n+1
				self.Loads.Next # move to the next load in the system

			# add graph
			self.add_graph()

			LogUtil.logger.debug('Successfully executed load method')
		except:
			LogUtil.exception_handler()

#=======================================================================================================================
	def _getLoads(self):
		"""Get the load setting for every load in the system. Please note that this 
		is not the actual load consumed. To get that you need a meter at the load bus.
		All values are reported in KW and KVar"""
		try:
			S={'P':{},'Q':{}}
			self.Loads.First # start with the first load in the list
			for n in range(0,self.Loads.Count):
				S['P'][self.Loads.Name],S['Q'][self.Loads.Name] = self.Loads.kW,self.Loads.kvar
				self.Loads.Next # move to the next load in the system

			LogUtil.logger.debug('Successfully executed _getLoads method')
			return S
		except:
			LogUtil.exception_handler()

#=======================================================================================================================
	def setV(self,Vpu,Vang=0,pccName='Vsource.source'):
		try:
			self._changeObj([[pccName,'pu',Vpu,'set'],[pccName,'angle',Vang,'set']])
			LogUtil.logger.debug('Successfully executed setV method')
		except:
			LogUtil.exception_handler()

#=======================================================================================================================
	def runpf(self, pccName='Vsource.source'):
		try:
			self.Solution.Solve()
			if self.Solution.Converged:
				self.Circuit.SetActiveElement(pccName)
				# -ve sign convention
				P,Q=self.Circuit.ActiveCktElement.SeqPowers[2]*-1,\
				self.Circuit.ActiveCktElement.SeqPowers[3]*-1
			else:
				P,Q=None,None

			LogUtil.logger.debug('Successfully executed runpf method')
			return P,Q,self.Solution.Converged
		except:
			LogUtil.exception_handler()

#=======================================================================================================================
	def runpf_qsts(self, datetimeObj,source='transmission',scale=1.0,pccName='Vsource.source'):
		"""
		Input:
		source can be either transmission or distribution
		scale -- scales the loadshape [0,1] to [0,1]*scale
		Sample call: df=dss.runpf_qsts(datetime.datetime(2021,1,22))
		"""
		try:
			loadShape=self.get_loadshape_data(datetimeObj,source)
			loadShape=loadShape*scale
			dt=datetime.timedelta(minutes=60)
			count=0
			for thisDispatchScaling in loadShape:
				self.scale_load(scale=thisDispatchScaling)
				self.runpf()
				assert self.Solution.Converged
				if count==0:
					df=self.get_df(t=datetimeObj+dt*count,scaling=thisDispatchScaling)
				else:
					df=pd.concat([df,self.get_df(t=datetimeObj+dt*count,\
					scaling=thisDispatchScaling)],ignore_index=True)
				count+=1
			return df
		except:
			LogUtil.exception_handler()


#=======================================================================================================================
	def get_line_currents(self,returnDf=False,returnDataAsJsonSerializeable=True,runpf=False):
		try:
			if runpf:
				self.runpf()# run powerflow
			self.Text.Command=f'export Currents ./line_currents.csv'
			df=pd.read_csv('./line_currents.csv')
			df=df.loc[df.Element.str.startswith('Line'),df.columns[0:7]]
			df=df.rename(columns={entry:entry.strip() for entry in df.columns})
			os.system('rm ./line_currents.csv')

			res={}
			res['data']=np.zeros((df.shape[0],6)) if not returnDataAsJsonSerializeable else {}
			for n in range(1,7):
				if not returnDataAsJsonSerializeable:
					res['data'][:,n-1]=df[df.columns[n]].values
				else:
					res['data'][df.columns[n].strip()]=df[df.columns[n]].values.tolist()
			res['line_name']=df.Element.values.tolist()

			if returnDf:
				res['df']=df

			return res
		except:
			LogUtil.exception_handler()

#=======================================================================================================================
	def _changeObj(self,objData):
		"""set/get an object property.
		Input: objData should be a list of lists of the format,
		[[objName,objProp,objVal,flg],...]

		objName -- name of the object.
		objProp -- name of the property.
		objVal -- val of the property. If flg is set as 'get', then objVal is not used.
		flg -- Can be 'set' or 'get'

		P.S. In the case of 'get' use a value of 'None' for objVal. The same object i.e.
		objData that was passed in as input will have the result i.e. objVal will be
		updated from 'None' to the actual value.

		Sample call: self._changeObj([['PVsystem.pv1','kVAr',25,'set']])
		self._changeObj([['PVsystem.pv1','kVAr','None','get']])
		"""
		try:
			for entry in objData:
				self.Circuit.SetActiveElement(entry[0])# make the required element as active element

				if entry[-1]=='set':
					self.CktElement.Properties(entry[1]).Val=entry[2]
				elif entry[-1]=='get':
					entry[2]=self.CktElement.Properties(entry[1]).Val

			LogUtil.logger.debug('Successfully executed _changeObj method')
		except:
			LogUtil.exception_handler()

#=======================================================================================================================
	def get_voltage(self,vtype='actual',busID=None,complexResult=True,returnArrayinYbusNodeOrder=False):
		"""Needs to be called after solve. Gathers Voltage (complex) of all buses into a,b,c phases
		and populates a dictionary and returns it."""
		try:
			Voltage={}
			entryMap={'1':'a','2':'b','3':'c'}

			if not busID:
				for n in range(0,self.Circuit.NumBuses):
					Voltage[self.Circuit.Buses(n).Name]={}
					if vtype=='actual':
						V=self.Circuit.Buses(n).Voltages
					elif vtype=='pu':
						V=self.Circuit.Buses(n).puVoltages

					count=0
					for entry in self.Circuit.Buses(n).Nodes:
						if complexResult:
							Voltage[self.Circuit.Buses(n).Name][entryMap[str(int(entry))]]=\
							V[count]+1j * V[count+1]
						else:
							Voltage[self.Circuit.Buses(n).Name][entryMap[str(int(entry))]]=\
							[V[count],V[count+1]]
						count+=2
			else:
				assert isinstance(busID,list) or isinstance(busID,tuple)
				for entry in busID:
					Voltage[entry]={}
					if vtype=='actual':
						V=self.Circuit.Buses(self.busname2ind[entry]).Voltages
					elif vtype=='pu':
						V=self.Circuit.Buses(self.busname2ind[entry]).puVoltages

					count=0
					for item in self.Circuit.Buses(self.busname2ind[entry]).Nodes:
						if complexResult:
							Voltage[entry][entryMap[str(int(item))]]=\
							V[count]+1j*V[count+1]
						else:
							Voltage[entry][entryMap[str(int(item))]]=\
							[V[count],V[count+1]]
						count+=2

			# returnArrayinYbusNodeOrder
			if returnArrayinYbusNodeOrder:
				nodeOrder=self.Circuit.YNodeOrder
				nodeOrderV=[Voltage[thisNodePhase.split('.')[0].lower()][entryMap[thisNodePhase.split('.')[1]]] \
				for thisNodePhase in nodeOrder]
				Voltage=np.array(nodeOrderV)

			LogUtil.logger.debug('Successfully executed get_voltage method')
			return Voltage
		except:
			LogUtil.exception_handler()

#=======================================================================================================================
	def get_voltage_magnitude(self,vtype='actual',busID=None):
		try:
			vmag={}
			phaseMap={1:'a',2:'b',3:'c'}

			nameFunc=self.Circuit.AllNodeNamesByPhase
			if vtype.lower()=='actual':
				valFunc=self.Circuit.AllNodeVmagByPhase
			elif vtype.lower()=='pu':
				valFunc=self.Circuit.AllNodeVmagPUByPhase

			for n in range(1,3+1):
				thisPhase=phaseMap[n]
				for thisNode,thisVal in zip(nameFunc(n),valFunc(n)):
					thisNodeName=thisNode.split('.')[0]
					if thisNodeName not in vmag:
						vmag[thisNodeName]={}
					vmag[thisNodeName][thisPhase]=thisVal

			LogUtil.logger.debug('Successfully executed get_voltage_magnitude method')
			return vmag
		except:
			LogUtil.exception_handler()

#=======================================================================================================================
	def add_der(self,conf):
		"""Models DER as constant power loads
		Sample call: dss.add_der(conf=[{'kw':30,'kvar':15,'bus':'670'}])"""
		try:
			threePhaseNodes=self._find_three_phase_nodes()
			Vact=self.get_voltage()
			Vpu=self.get_voltage('pu')
			phaseMap={'a':'1','b':'2','c':'3'}

			default={'vminpu':0.85,'vmaxpu':1.2,'conn':'Wye'}

			for entry in conf:
				thisConf=copy.deepcopy(default)
				thisConf.update(entry)

				if 'pmax' not in thisConf:
					thisConf['pmax']=thisConf['kw']
				if 'pmin' not in thisConf:
					thisConf['pmin']=0
				if 'qmax' not in thisConf:
					thisConf['qmax']=thisConf['kvar']
				if 'qmin' not in thisConf:
					thisConf['qmin']=0

				if thisConf['bus'] in Vact:
					nPhase=len(Vact[thisConf['bus']])
					self.dermap[thisConf["bus"]]={}
					for thisPhase in Vact[thisConf['bus']]:
						thisV=(abs(Vact[thisConf['bus']][thisPhase])/abs(Vpu[thisConf['bus']][thisPhase]))*1e-3
						directive=f'New Load.{thisConf["bus"]}_der_{thisPhase} '
						directive+=f'Bus1={thisConf["bus"]}.{phaseMap[thisPhase]} '
						directive+=f'Phases=1 Conn={thisConf["conn"]} Model=1 kV={thisV} '
						# equally split power across phases
						directive+=f'kW=-{thisConf["kw"]/nPhase} kvar=-{thisConf["kvar"]/nPhase} '
						directive+=f'vminpu={thisConf["vminpu"]} vmaxpu={thisConf["vmaxpu"]}'
						self.Text.Command=directive
						self.dermap[thisConf["bus"]][thisPhase]=\
						{prop:thisConf[prop] for prop in ['kw','kvar','pmax','pmin','qmax','qmin']}
						self.dermap[thisConf["bus"]][thisPhase]['id']=f'Load.{thisConf["bus"]}_der_{thisPhase}'
						self.dermap[thisConf["bus"]][thisPhase]['kw']=self.dermap[thisConf["bus"]][thisPhase]['kw']/nPhase
						self.dermap[thisConf["bus"]][thisPhase]['kvar']=self.dermap[thisConf["bus"]][thisPhase]['kvar']/nPhase
						self.dermap[thisConf["bus"]][thisPhase]['nodeid']=thisConf['bus']
						self.dermap[thisConf["bus"]][thisPhase]['phase']=thisPhase
				else:
					print(f'Requested bus {entry["bus"]} is not found')
					LogUtil.logger.warning(f'Requested bus {entry["bus"]} is not found')

			LogUtil.logger.debug('Successfully executed add_der method')
		except:
			LogUtil.exception_handler()

#=======================================================================================================================
	def set_der(self,conf):
		try:
			setpoint=[]
			for busID in conf:
				for thisPhase in conf[busID]:
					for prop in conf[busID][thisPhase]:
						setpoint.append([self.dermap[busID][thisPhase]['id'],prop,conf[busID][thisPhase][prop],'set'])
						self._changeObj(setpoint)
						if busID in self.dermap and thisPhase in self.dermap[busID] and prop in self.dermap[busID][thisPhase]:
							self.dermap[busID][thisPhase][prop]=conf[busID][thisPhase][prop]
							if prop.lower()=='kw' or prop.lower()=='kvar':
								self.dermap[busID][thisPhase][prop]=-self.dermap[busID][thisPhase][prop]

			LogUtil.logger.debug('Successfully executed set_der method')
		except:
			LogUtil.exception_handler()

#=======================================================================================================================
	def _find_three_phase_nodes(self):
		try:
			V=self.get_voltage()
			threePhaseNodes=[thisNode for thisNode in V if len(V[thisNode])==3]

			LogUtil.logger.debug('Successfully executed _find_three_phase_nodes method')
			return threePhaseNodes
		except:
			LogUtil.exception_handler()

#=======================================================================================================================
	def get_sensitivity(self,eps=1):
		try:
			res={}
			for busID in self.dermap:
				res[busID]={}
				for thisPhase in self.dermap[busID]:
					res[busID][thisPhase]={}
					P,Q,converged=self.runpf()
					Vpu=self.get_voltage('pu')
					self.set_der({busID:{thisPhase:{'kvar':-self.dermap[busID][thisPhase]['kvar']-eps}}})
					P_,Q_,converged=self.runpf()
					Vpu_=self.get_voltage('pu')
					res[busID][thisPhase]['dvdq']={}
					for thisBus in Vpu:
						res[busID][thisPhase]['dvdq'][thisBus]={}
						for nodePhase in Vpu[thisBus]:
							res[busID][thisPhase]['dvdq'][thisBus][nodePhase]=\
							(abs(Vpu_[thisBus][nodePhase])-abs(Vpu[thisBus][nodePhase]))/eps
					self.set_der({busID:{thisPhase:{'kvar':-self.dermap[busID][thisPhase]['kvar']+eps}}})# reset

			LogUtil.logger.debug('Successfully executed get_sensitivity method')
			return res
		except:
			LogUtil.exception_handler()
#=======================================================================================================================
	def get_data_driven_sensitivity(self,eps=1):
		try:
			res={}
			for busID in self.dermap: 
				res[busID]={}
				for thisPhase in self.dermap[busID]: 
					res[busID][thisPhase]={}
					P,Q,converged=self.runpf() 
					Vpu=self.get_voltage('pu') 
					self.set_der({busID:{thisPhase:{'kvar':-self.dermap[busID][thisPhase]['kvar']-eps}}}) 
					
					self.scale_load(0.5 + np.random.rand(1))

					P_,Q_,converged=self.runpf()	
					Vpu_=self.get_voltage('pu')	
					res[busID][thisPhase]['dvdq']={}
					for thisBus in Vpu: 
						res[busID][thisPhase]['dvdq'][thisBus]={}
						for nodePhase in Vpu[thisBus]:
							# compute impedance
							res[busID][thisPhase]['dvdq'][thisBus][nodePhase]=\
							((abs(Vpu_[thisBus][nodePhase])-abs(Vpu[thisBus][nodePhase]))/eps)*abs(Vpu[busID][thisPhase])
					self.set_der({busID:{thisPhase:{'kvar':-self.dermap[busID][thisPhase]['kvar']+eps}}})# reset


			LogUtil.logger.debug('Successfully executed get_sensitivity method')
			return res
		except:
			LogUtil.exception_handler()

#=======================================================================================================================
	def get_control_setpoints(self,optimMode=1,vmin=0.9,vmax=1.1):
		try:
			# do a lazy import. If import fails, log the msg and raise an error.
			import qpsolvers as qp

			self._assign_var_index(optimMode)

			self.P=P=np.eye(self.varInd['nVars'])*2
			q=np.zeros(self.varInd['nVars'])
			if optimMode==2:
				val=np.ones(self.varInd['nVars'])*1e2 #### TODO: normalize and use weightage based on sensitivity
				for thisVar in self.varInd['name2ind']:
					if 'xu_' not in thisVar:
						val[self.varInd['name2ind'][thisVar]]=1e0
				np.fill_diagonal(P,val)

			dvdq=self.get_sensitivity()
			V_=self.get_voltage_magnitude(vtype='pu')
			
			self.G=G=np.zeros((len(self.varInd['voltageIndex']),self.varInd['nVars']))
			count=0; V=[]
			for thisNode in V_:
				for thisPhase in V_[thisNode]:
					V.append(V_[thisNode][thisPhase])
					for thisControllerVarInd in range(self.varInd['nControlVars']):
						thisObj=self.varInd['id2obj'][self.varInd['ind2name'][thisControllerVarInd]]
						controllerNodeID,controllerPhase=thisObj['nodeid'],thisObj['phase']
						G[count,thisControllerVarInd]=\
						dvdq[controllerNodeID][controllerPhase]['dvdq'][thisNode][thisPhase]
					count+=1

			if isinstance(vmin,float):
				vmin=np.array([vmin]*len(V))
			if isinstance(vmax,float):
				vmax=np.array([vmax]*len(V))
			self.h=h=vmax-V

			reqChange=0.01
			if optimMode==1:
				h[31]=-reqChange
				h[32]=-reqChange
				h[33]=-reqChange
				A=None
				b=None
			elif optimMode==2:
				self.A=A=np.zeros((self.varInd['nVars']-self.varInd['nControlVars'],self.varInd['nVars']))
				self.b=b=np.zeros((self.varInd['nVars']-self.varInd['nControlVars']))
				count=0
				for thisNode in V_:
					if len(V_[thisNode])>1:
						for thisPhase in V_[thisNode]:
							for thisControllerVarInd in range(self.varInd['nControlVars']):
								thisObj=self.varInd['id2obj'][self.varInd['ind2name'][thisControllerVarInd]]
								controllerNodeID,controllerPhase=thisObj['nodeid'],thisObj['phase']
								A[count,thisControllerVarInd]=\
								-dvdq[controllerNodeID][controllerPhase]['dvdq'][thisNode][thisPhase]
							A[count,self.varInd['name2ind'][f'xv_{thisNode}_{thisPhase}']]=1
							b[count]=V_[thisNode][thisPhase]
							count+=1

				for thisNode in V_:
					if len(V_[thisNode])==2:
						phases=list(V_[thisNode].keys())
						A[count,self.varInd['name2ind'][f'xv_{thisNode}_{phases[0]}']]=1
						A[count,self.varInd['name2ind'][f'xv_{thisNode}_{phases[1]}']]=-1
						A[count,self.varInd['name2ind'][f'xu_{thisNode}_1']]=1
						count+=1
					elif len(V_[thisNode])==3:
						phases=list(V_[thisNode].keys())
						A[count,self.varInd['name2ind'][f'xv_{thisNode}_{phases[0]}']]=1
						A[count,self.varInd['name2ind'][f'xv_{thisNode}_{phases[1]}']]=-1
						A[count,self.varInd['name2ind'][f'xu_{thisNode}_1']]=1
						count+=1
						A[count,self.varInd['name2ind'][f'xv_{thisNode}_{phases[1]}']]=1
						A[count,self.varInd['name2ind'][f'xv_{thisNode}_{phases[2]}']]=-1
						A[count,self.varInd['name2ind'][f'xu_{thisNode}_2']]=1
						count+=1

			# solve
			x=qp.quadprog_solve_qp(P=P,q=q,G=G,h=h,A=A,b=b)
			res={'x':x,'success':True} if not isinstance(x,type(None)) else {'x':x,'success':False}
			LogUtil.logger.info('Quadprog success flag: f{res["success"]}')

			# validate
			Vnc=self.get_voltage_magnitude('pu')
			conf={}
			for thisControllerVarInd in range(self.varInd['nControlVars']):
				thisObj=self.varInd['id2obj'][self.varInd['ind2name'][thisControllerVarInd]]
				controllerNodeID,controllerPhase=thisObj['nodeid'],thisObj['phase']
				if controllerNodeID not in conf:
					conf[controllerNodeID]={}
				conf[controllerNodeID][controllerPhase]={'kvar':-res['x'][thisControllerVarInd]}

			self.set_der(conf)
			self.runpf()
			Vc=self.get_voltage_magnitude('pu')

			for thisControllerVarInd in range(self.varInd['nControlVars']):
				thisObj=self.varInd['id2obj'][self.varInd['ind2name'][thisControllerVarInd]]
				controllerNodeID,controllerPhase=thisObj['nodeid'],thisObj['phase']
				print(f'Node:{controllerNodeID} Phase:{controllerPhase} pre->{Vnc[controllerNodeID][controllerPhase]} '+\
				f'post->{Vc[controllerNodeID][controllerPhase]}')
			print(f'cost is : {(np.diag(P)*res["x"]**2).sum()}')
			self.cost=np.diag(P)*res["x"]**2

			return res
		except:
			LogUtil.exception_handler()

#=======================================================================================================================
	def _assign_var_index(self,optimMode=1):
		try:
			# assign voltage index
			V=self.get_voltage_magnitude(vtype='pu')
			self.varInd={'name2ind':{},'ind2name':{},'id2obj':{},'voltageIndex':{},'node2voltageIndex':{},\
			'phaseControlIndex':{}}
			
			count=0
			for thisNode in V:
				self.varInd['node2voltageIndex'][thisNode]=[]
				for thisPhase in V[thisNode]:
					self.varInd['voltageIndex'][count]={'node':thisNode,'phase':thisPhase}
					self.varInd['node2voltageIndex'][thisNode].append(count)
					count+=1

			count=0
			for thisControlNode in self.dermap:
				for thisController in self.dermap[thisControlNode]:
					thisID=self.dermap[thisControlNode][thisController]['id']
					self.varInd['name2ind'][thisID]=count
					self.varInd['ind2name'][count]=thisID
					self.varInd['id2obj'][thisID]=self.dermap[thisControlNode][thisController]
					count+=1
			self.varInd['nControlVars']=count
			if optimMode==2:
				for thisNode in V:
					if len(V[thisNode])>1:
						for thisPhase in V[thisNode]:
							self.varInd['name2ind'][f'xv_{thisNode}_{thisPhase}']=count
							self.varInd['ind2name'][count]=f'xv_{thisNode}_{thisPhase}'
							count+=1
				for thisNode in V:
					if len(V[thisNode])==2:
						self.varInd['name2ind'][f'xu_{thisNode}_1']=count
						self.varInd['ind2name'][count]=f'xu_{thisNode}_1'
						count+=1
					elif len(V[thisNode])==3:
						self.varInd['name2ind'][f'xu_{thisNode}_1']=count
						self.varInd['ind2name'][count]=f'xu_{thisNode}_1'
						count+=1
						self.varInd['name2ind'][f'xu_{thisNode}_2']=count
						self.varInd['ind2name'][count]=f'xu_{thisNode}_2'
						count+=1


			self.varInd['nVars']=count

		except:
			LogUtil.exception_handler()

#=======================================================================================================================
	def get_loadshape_data(self,datetimeObj,source='transmission',normalize=True):
		"""Source can be transmission or distribution"""
		try:
			nDay=datetimeObj.timetuple().tm_yday
			Pd=self.loadshape['Pd'][(nDay-1)*24:nDay*24]
			Pd=np.array(Pd)
			if normalize:
				Pd=Pd/Pd.max()

			return Pd
		except:
			LogUtil.exception_handler()

#=======================================================================================================================
	def scale_load(self,scale=None,loadScaleMap=None,resetToBaseLoadingBeforeScaling=False):
		"""Sets the load shape by scaling each load in the system with a scaling
			factor scale.
			Input: scale -- A scaling factor for loads such that P+j*Q=scale*(P0+j*Q0)
			P.S. loadShape should be called at every dispatch i.e. only load(t) is set.
			"""
		try:
			self.Loads.First # start with the first load in the list
			if not loadScaleMap:
				for n in range(self.Loads.Count):
					if '_der' not in self.Loads.Name:
						self.Loads.kW=self.S0['P'][self.Loads.Name]*scale
						self.Loads.kvar=self.S0['Q'][self.Loads.Name]*scale
						self.Loads.Next # move to the next load in the system
			else:
				if resetToBaseLoadingBeforeScaling:
					for n in range(self.Loads.Count):
						if '_der' not in self.Loads.Name:
							self.Loads.kW=self.S0['P'][self.Loads.Name]
							self.Loads.kvar=self.S0['Q'][self.Loads.Name]
							self.Loads.Next # move to the next load in the system
				for thisLoadName in loadScaleMap:
					if '_der' not in thisLoadName:
						thisIdx=self.loadname2ind[thisLoadName]
						thisScale=loadScaleMap[thisLoadName]
						self.Loads.idx=thisIdx
						self.Loads.kW=self.S0['P'][thisLoadName]*thisScale
						self.Loads.kvar=self.S0['Q'][thisLoadName]*thisScale
		except:
			LogUtil.exception_handler()


#=======================================================================================================================
	def get_df(self,t=1,scenario=None,scaling=None,addLineCurrents=True,includeAssetData=False):
		"""Sample call: df=pm.get_df(datetime.datetime(2021,1,4,1,0),'control')"""
		try:
			vmag_pu=self.get_voltage_magnitude('pu')
			vmag_actual=self.get_voltage_magnitude('actual')

			res={'element':[],'property':[],'value':[]}
			if includeAssetData and self.assetData:
				res['uuid']=[]
				res['lat_lon']=[]

			S=self._getLoads()
			for thisNode in vmag_actual:
				for thisPhase in vmag_actual[thisNode]:
					res['element'].append(thisNode)
					res['property'].append('vmag_actual_'+thisPhase)
					res['value'].append(vmag_actual[thisNode][thisPhase])
					if includeAssetData and self.assetData:
						res['uuid'].append(self.assetData['uuid'][thisNode])
						res['lat_lon'].append(self.assetData['lat_lon'][thisNode])
					res['element'].append(thisNode)
					res['property'].append('vmag_pu_'+thisPhase)
					res['value'].append(vmag_pu[thisNode][thisPhase])
					if includeAssetData and self.assetData:
						res['uuid'].append(self.assetData['uuid'][thisNode])
						res['lat_lon'].append(self.assetData['lat_lon'][thisNode])

			# add current injection measurement
			if addLineCurrents:
				m=self.get_line_currents(returnDf=True)
				df_m=m['df']
				nMeasurements=df_m.shape[0]
				headerMap={'imag_actual_a':'I1_1','imag_actual_b':'I1_2','imag_actual_c':'I1_3',\
					'iang_actual_a':'Ang1_1','iang_actual_b':'Ang1_2','iang_actual_c':'Ang1_3'}

				for thisProp in headerMap:
					res['element']+=df_m.Element.values.tolist()
					res['property']+=[thisProp]*nMeasurements
					res['value']+=df_m[headerMap[thisProp]].values.tolist()
					if includeAssetData and self.assetData:
						for thisEl in df_m.Element.values.tolist():
							res['uuid'].append(self.assetData['uuid'][thisEl])
							res['lat_lon'].append(self.assetData['lat_lon'][thisEl])

			if t:
				res['t']=[t]*len(res['element'])
			if scenario:
				res['scenario']=[scenario]*len(res['element'])
			if scaling:
				res['scaling']=[scaling]*len(res['element'])
			df=pd.DataFrame(res)

			return df
		except:
			LogUtil.exception_handler()

#=======================================================================================================================
	def updates_to_distributed_controller(self):
		try:
			res={}
			_=self.runpf()
			res['vmag_pu']=self.get_voltage_magnitude('pu')
			res['vmag_actual']=self.get_voltage_magnitude('actual')
			res['sensitivity']=self.get_sensitivity()
			return res
		except:
			LogUtil.exception_handler()

#=======================================================================================================================
	def updates_from_distributed_controller(self):
		try:
			pass
		except:
			LogUtil.exception_handler()

#=======================================================================================================================
	def get_admittance_matrix(self,fpath=None,makeSparse=False):
		"""fpath should point to dss file where load definitions are removed and control mode
		should be set to off. Returns the res['ybus'] such that S=V.*conj(ybus*V) where order of the nodes
		are given in res['nodeOrder'].
		Sample call: Ybus=dss.get_admittance_matrix(dss.currentFpath,True)"""
		try:
			res={}
			if fpath:
				self.load_case(fpath)
			self.runpf()
			ybus_=self.Circuit.SystemY
			ybus_.dtype='complex'
			res['nodeOrder']={'names':self.Circuit.YNodeOrder}
			res['nodeOrder']['name2ind']={res['nodeOrder']['names'][n]:n for n in range(len(res['nodeOrder']['names']))}
			res['nodeOrder']['ind2name']={n:res['nodeOrder']['names'][n] for n in range(len(res['nodeOrder']['names']))}
			res['ybus']=np.reshape(ybus_,(len(res['nodeOrder']['names']),len(res['nodeOrder']['names'])))

			if makeSparse:# coo format
				r,c=np.where(res['ybus'])
				v=res['ybus'][r,c]
				res['ybus']={'r':r.tolist(),'c':c.tolist(),'v':v.tolist()}

			return res
		except:
			LogUtil.exception_handler()

#=======================================================================================================================
	def adjust_operating_point(self,pd,qd,savepath=None):
		try:
			# adjust load
			nMatches=0
			self.Loads.First # start with the first load in the list
			for n in range(0,self.Loads.Count):
				if self.Loads.Name in pd and self.Loads.Name in qd:
					self.Loads.kW=pd[self.Loads.Name]
					self.Loads.kvar=qd[self.Loads.Name]
					nMatches+=1
				self.Loads.Next # move to the next load in the system

			if savepath:
				res=self.get_optim_data(savepath=savepath)
			LogUtil.logger.info(f'Successfully executed adjust_operating_point method with {nMatches} changes')
		except:
			LogUtil.exception_handler()

#=======================================================================================================================
	def get_optim_data(self,fpath=None,disableElements=True,savepath=None,usePdQd=True):
		"""Gets the required data for distribution optimization.
		Sample call: res=dss.get_optim_data(dss.currentFpath,disableElements=True,savepath='distcase13_opf_data.json')
		"""
		try:
			res=self.get_admittance_matrix(fpath=fpath,makeSparse=True)
			res_pf=self.runpf()
			LogUtil.logger.info(f'power flow: {res_pf}')
			V=self.get_voltage(returnArrayinYbusNodeOrder=True)
			Vpu=self.get_voltage('pu',returnArrayinYbusNodeOrder=True)
			Vbase=abs(V)/abs(Vpu)
			res['x0']=abs(V).tolist()+np.angle(V).tolist()

			res['ybus']['vr']=np.array(res["ybus"]["v"]).real.tolist()
			res['ybus']['vi']=np.array(res["ybus"]["v"]).imag.tolist()
			res['ybus'].pop('v')
			assert len(res['ybus']['vr'])==len(res['ybus']['vi'])==len(res['ybus']['r'])==len(res['ybus']['c'])

			# fixed cost for source bus
			res['gencost']={"genid":[1,2,3],\
				"c2":[0.11,0.11,0.11],\
				"c1":[5,5,5],"c0":[0,0,0]}

			# add solar
			pg0=[0,0,0]; qg0=[0,0,0]
			if self.dermap:
				res['solar']=self.dermap
				for entry in self.dermap:
					for item in self.dermap[entry]:
						res['gencost']['genid'].append(res['gencost']['genid'][-1]+1)
						res['gencost']['c0'].append(0)
						res['gencost']['c1'].append(-0.1*1e-3)
						res['gencost']['c2'].append(0)
						pg0.append(self.dermap[entry][item]['kw']*1e3)# in watts
						qg0.append(self.dermap[entry][item]['kvar']*1e3) # in var
			res['x0'].extend(pg0)
			res['x0'].extend(qg0)

			res['pd']=self.S0['P']
			res['qd']=self.S0['Q']

			count=0
			pd=[0]*len(res['nodeOrder']['name2ind'])
			qd=[0]*len(res['nodeOrder']['name2ind'])
			phaseMap={'1':'a','2':'b','3':'c'}
			if usePdQd:
				for thisNode in res['nodeOrder']['name2ind']:
					thisNodeName,thisNodePhase=thisNode.split('.')
					if thisNodePhase in phaseMap:
						if thisNodeName+phaseMap[thisNodePhase] in res['pd']:
							pd[count]=res['pd'][thisNodeName+phaseMap[thisNodePhase]]*1e3
							qd[count]=res['qd'][thisNodeName+phaseMap[thisNodePhase]]*1e3
						elif thisNodeName in res['pd']:
							nPhase=0
							if thisNodeName+'.1' in res['nodeOrder']['name2ind']:
								nPhase+=1
							if thisNodeName+'.2' in res['nodeOrder']['name2ind']:
								nPhase+=1
							if thisNodeName+'.3' in res['nodeOrder']['name2ind']:
								nPhase+=1
							pd[count]=(res['pd'][thisNodeName]*1e3)/nPhase
							qd[count]=(res['qd'][thisNodeName]*1e3)/nPhase
					count+=1

			res['pd']=pd
			res['qd']=qd
			res['vbase']=Vbase.tolist()

			if savepath:
				if os.path.basename(savepath)==savepath:# use data folder to store if abspath is not given
					savepath=os.path.join(self.baseDir,'data','opf_data',savepath)
				json.dump(res,open(savepath,'w'))

			return res
		except:
			LogUtil.exception_handler()

#=======================================================================================================================
	def disable_elements(self,regcontrol=True,vsource=True,isource=True,load=True,generator=True,
	pvsystem=True,storage=True,capacitor=True,setTap=None):
		try:
			if regcontrol:
				self.Text.Command='batchedit regcontrol..* enabled=false'
			if vsource:
				self.Text.Command='batchedit vsource..* enabled=false'
			if isource:
				self.Text.Command='batchedit isource..* enabled=false'
			if load:
				self.Text.Command='batchedit load..* enabled=false'
			if generator:
				self.Text.Command='batchedit generator..* enabled=false'
			if pvsystem:
				self.Text.Command='batchedit pvsystem..* enabled=false'
			if storage:
				self.Text.Command='batchedit storage..* enabled=false'
			if capacitor:
				self.Text.Command='batchedit capacitor..* enabled=false'
			if setTap:
				self.Text.Command=f'batchedit transformer..* wdg=2 tap={setTap}'
			self.Text.Command='CalcVoltageBases'
			self.Text.Command='buildY'
		except:
			LogUtil.exception_handler()

#=======================================================================================================================
	def add_graph(self):
		try:
			self.Lines.First
			edges=[[self.Lines.Bus1.split(".")[0],self.Lines.Bus2.split(".")[0]]]
			while self.Lines.Next!=0:
				edges.append([self.Lines.Bus1.split(".")[0],self.Lines.Bus2.split(".")[0]])
			self.G=nx.Graph()
			self.G.add_edges_from(edges)
		except:
			LogUtil.exception_handler()

#=======================================================================================================================
	def show_graph(self):
		try:
			dotPath=os.path.join(os.path.dirname(self.currentFpath),'dssGraph.dot')
			svgPath=os.path.join(os.path.dirname(self.currentFpath),'dssGraph.svg')

			res=self.get_admittance_matrix(makeSparse=True)
			r=np.array(res['ybus']['r'])
			c=np.array(res['ybus']['c'])
			names=np.array(res['nodeOrder']['names'])

			res={}
			for n in range(max(r)+1):
				fromBus=names[n].split('.')[0]
				if not fromBus in res:
					res[fromBus]=[]
				res[fromBus].extend(list(set([item.split('.')[0] for item in names[c[r==n]].tolist()]).difference([fromBus])))
				res[fromBus]=list(set(res[fromBus]))

			processed={}
			dataStr='graph{\nsplines=True'
			for thisNode in res:
				for item in res[thisNode]:
					if (thisNode,item) not in processed and (item,thisNode) not in processed:
						processed[(thisNode,item)]=[]
						dataStr+=f'"{thisNode}"--"{item}"\n'
			dataStr+='}'

			f=open(dotPath,'w')
			f.write(dataStr)
			f.close()

			os.system(f'dot -Tsvg {dotPath} -o {svgPath} && xdg-open {svgPath}')
		except:
			LogUtil.exception_handler()


#=======================================================================================================================
if __name__ == '__main__':
	dss=DistModel(serverMode=True)
	comm_end=False
	while not comm_end:
		comm_end=dss.msg2procedure()
	dss.server_close()


