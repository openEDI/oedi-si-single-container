import os

from oct2py import octave


def algorithm(inputData,logger):
	"""
	Mock DOPF Algorithm to show how to interface external algorithm with oedisi.
	inputData and outputData are dictionaries with the keys defined below. P.S. This function
	is called at every time step.

	Input: inputData -- Topology, VoltagesMagnitude, PowersReal and PowersImaginary
	Topology -- Admittance Matrix,...
	PowersReal -- values (list), ids (list), units
	PowersImaginary -- values (list), ids (list), units
	logger -- logger to log information to a file. See usage below.

	Output: outputData --
	"""

	baseDir=os.path.dirname(os.path.abspath(__file__))

	load_P=inputData['PowersReal']['values']
	load_Q=inputData['PowersImaginary']['values']
	load_id=inputData['PowersReal']['ids']

	#### TODO: Alg developers should replace DOPF_timeseries to run with individual time step data
	#### currently DOPF_timeseries runs for 96 dispatches each time this function is called, instead
	#### the algorithm should only optimize for each dispatch and then move to the next. If multi-period
	#### optimization is needed then the algorithm should work with forecasted data or collect actual data,
	#### in which case, this file can be modified to include a conditional clause where DOPF_timeseries()
	#### is run only at the last time step -- similar to the original code.
	#### load_P.npy,load_Q.npy should be written to disk based on the data provided here. Use,
	#### np.save(os.path.join(baseDir,'data','load_P.npy'),load_P) and
	#### np.save(os.path.join(baseDir,'data','load_Q.npy'),load_Q)

	output = octave.DOPF_timeseries(load_P, load_Q, load_id,\
		os.path.join(baseDir,'master.dss'),os.path.join(baseDir,'IEEE123_input.csv'))

	# example usage of logger
	logger.info(f'output from DOPF_timeseries::::{output["x"].shape},{output}')

	outputData=inputData #### TODO: Alg developers should replace this with actual output data

	return outputData


