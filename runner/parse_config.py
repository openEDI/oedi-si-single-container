import os
import json
import pdb


if __name__=="__main__":
	baseDir=os.path.dirname(os.path.abspath(__file__))
	availableFederates=json.load(open(os.path.join(baseDir,'available_federates.json')))
	userConfig=json.load(open(os.path.join(baseDir,'user_config.json')))
	unavailableFed=set(userConfig['federates']).difference(availableFederates)
	assert not unavailableFed,f"federates {unavailableFed} are unavailable"

	# create config_runner
	nFed=len(userConfig['federates'])+len(userConfig['private_federates'])
	availableFederates['broker']['exec']=f"helics_broker -f {nFed} --loglevel=debug"
	config={"name": "gadal_single_container_example",'federates':[]}

	for thisFed in userConfig['federates']:
		config['federates'].append(availableFederates[thisFed])

	if 'run_broker' in userConfig and userConfig['run_broker']:
		config['federates'].append(availableFederates['broker'])

	json.dump(config,open(os.path.join(baseDir,'config_runner.json'),'w'),indent=3)
