import copy

def algorithm(inputData):
	"""
	Mock DSSE Algorithm to show how to interface external algorithm with oedisi.
	inputData and outputData are dictionaries with the keys defined below.

	Input: inputData -- Topology, VoltagesMagnitude, PowersReal and PowersImaginary
	Topology -- Admittance Matrix,...
	VoltagesMagnitude -- values (list), ids (list), units
	PowersReal -- values (list), ids (list), units
	PowersImaginary -- values (list), ids (list), units

	Output: outputData -- VoltagesMagnitude and VoltagesAngle
	VoltagesMagnitude -- values (list), ids (list), units
	VoltagesAngle -- values (list), ids (list), units
	"""

	# simply return vmag and set vang=0
	outputData={}
	outputData['VoltagesMagnitude']=inputData['VoltagesMagnitude']
	outputData['VoltagesAngle']=copy.copy(inputData['VoltagesMagnitude'])
	outputData['VoltagesAngle']['values']=[0.0]*len(inputData['VoltagesMagnitude']['values'])

	return outputData
