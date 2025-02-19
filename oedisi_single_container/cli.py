import os
import sys
import uuid
import re
import json
import pdb
import subprocess
import shlex
from io import StringIO

import pandas as pd
import click


if 'runner' in os.listdir(os.path.dirname(os.path.abspath(__file__))):
	baseDir=os.path.dirname(os.path.abspath(__file__))
else:
	baseDir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
defaultConfigPath=os.path.join(baseDir,'runner','cli_default_config.json')
defaultConfig=json.load(open(defaultConfigPath))


if 'win32' in sys.platform or 'cygwin' in sys.platform:
	isWindows=True
	copyCmd='robocopy'
	def convert_windows_to_linux_path(windows_path):
		# Replace backslashes with forward slashes
		linux_path = windows_path.replace('\\', '/')
		
		# Convert drive letter to Linux-style path
		if linux_path[1] == ':':
			drive_letter = linux_path[0].lower()
			linux_path = f"/mnt/{drive_letter}" + linux_path[2:]
		
		return linux_path
else:
	isWindows=False
	copyCmd='cp'


@click.group()
def main():
	pass


@main.command()
@click.option("-p","--project_dir_path", required=True, help="Path to project directory")
@click.option("-c","--config", required=True, help="Path to config file")
@click.option("-r","--run_as_admin", required=False, default=False, help="Should docker be run as root")
@click.option("-t","--tag", required=False, default=defaultConfig['tag'], help="Image tag to be used")
@click.option("--podman", required=False, default=defaultConfig['podman'], help="Use podman instead of docker")
@click.option("-i","--interactive", required=False, default=False, help="Interactive mode")
def run(project_dir_path,config,run_as_admin,tag,podman,interactive):
	"""Runs the co-simulation"""
	project_dir_path=os.path.abspath(project_dir_path)
	config=os.path.abspath(config)
	containerEngine='podman' if podman else 'docker'

	directive=''
	if run_as_admin:
		directive+='sudo '

	name='singlecontainerapp_'+uuid.uuid4().hex

	if isWindows:
		windowsVolumeMount(baseDir,project_dir_path,config,tag,containerEngine)
		directive=f'{containerEngine} run --name {name} '+\
			f'-v oedisisc_runtime:/home/runtime '

		if interactive:
			directive+='-it --entrypoint /bin/bash '

		directive+=f'{tag}'
		print(f"Run directive:{directive}")
		os.system(directive)

		# copy output
		os.system(f'wsl -d podman-machine-default -u user enterns podman cp  {name}:/home/output '+\
			f'"{convert_windows_to_linux_path(project_dir_path)}"')
		os.system(f'{containerEngine} rm {name}')
	else:
		directive+=f'{containerEngine} run --rm --name {name} -v {os.path.join(baseDir,"runner")}:/home/runtime/runner '+\
			f'-v "{config}":/home/runtime/runner/user_config.json '+\
			f'-v "{os.path.join(project_dir_path,"user_federates")}":/home/runtime/user_federates '+\
			f'-v "{os.path.join(baseDir,"user_interface")}":/home/runtime/user_interface '+\
			f'-v "{os.path.join(project_dir_path,"output")}":/home/output '

		if interactive:
			directive+='-it --entrypoint /bin/bash '

		directive+=f'{tag}'
		os.system(directive)

def windowsVolumeMount(baseDir,project_dir_path,configPath,tag,containerEngine):

	# check for missing volumes
	missingVolumes=[]
	for entry in ['oedisisc_runtime']:
		err=os.system(f'{containerEngine} volume inspect {entry}')
		if err!=0:
			missingVolumes.append(entry)

	for entry in missingVolumes:
		err=os.system(f'{containerEngine} volume create {entry}')

	# mount
	err=os.system(f'{containerEngine} container create --name oedisisc_dummy -v oedisisc_runtime:/home/runtime {tag}')
	
	# copy	
	os.system(f'robocopy "{os.path.join(baseDir,"runner")}" "{os.path.join(project_dir_path,"runner")}" *.*')
	os.system(f'robocopy "{os.path.dirname(configPath)}" "{os.path.join(project_dir_path,"runner")}" '+	configPath.split("\\")[-1])	
	
	if containerEngine == "podman": #podman cp command doesn't work in Windows. This is a temporary workaround		
		err=os.system(f'wsl -d podman-machine-default -u user enterns podman cp "{convert_windows_to_linux_path(os.path.join(project_dir_path,"runner"))}" oedisisc_dummy:/home/runtime')
		err=os.system(f'wsl -d podman-machine-default -u user enterns podman cp "{convert_windows_to_linux_path(os.path.join(baseDir,"user_interface"))}" oedisisc_dummy:/home/runtime')
		err=os.system(f'wsl -d podman-machine-default -u user enterns podman cp "{convert_windows_to_linux_path(os.path.join(baseDir,"user_federates"))}" oedisisc_dummy:/home/runtime')
	else:
		err=os.system(f'{containerEngine} cp "{os.path.join(project_dir_path,"runner")}" oedisisc_dummy:/home/runtime')
		err=os.system(f'{containerEngine} cp "{os.path.join(baseDir,"user_interface")}" oedisisc_dummy:/home/runtime')
		err=os.system(f'{containerEngine} cp "{os.path.join(project_dir_path,"user_federates")}" oedisisc_dummy:/home/runtime')

	# delete
	err=os.system(f'{containerEngine} rm oedisisc_dummy')


@main.command(name="init")
@click.option("-p","--project_dir_path", required=True, help="Path to template folder")
def init(project_dir_path):
	"""Initializes a new project"""
	print(f"Project dir path:{project_dir_path}")
	print(f"Base dir path:{baseDir}")
	
	if isWindows:
		#err=os.system(f'mkdir {os.path.join(project_dir_path,"config")} {os.path.join(project_dir_path,"output")}') #These commands don't work if there are spaces
		#err=os.system(f'{copyCmd} {os.path.join(baseDir,"runner")} {os.path.join(project_dir_path,"config")}')
		#err=os.system(f'{copyCmd} {os.path.join(baseDir,"user_federates")} {os.path.join(project_dir_path,"user_federates")} /MIR')
		os.makedirs(os.path.join(project_dir_path, "config"), exist_ok=True) #These commands will work if there are spaces in the path names
		os.makedirs(os.path.join(project_dir_path, "output"), exist_ok=True)
		source_path = os.path.join(baseDir, "runner")
		destination_path = os.path.join(project_dir_path, "config")		
		copy_cmd = f'{copyCmd} "{source_path}" "{destination_path}"' # Construct the copy command with proper quoting
		err = os.system(copy_cmd)
		
		source_path = os.path.join(baseDir, "user_federates")
		destination_path = os.path.join(project_dir_path, "user_federates")		
		copy_cmd = f'{copyCmd} "{source_path}" "{destination_path}" /MIR' # Construct the copy command with proper quoting
		err = os.system(copy_cmd)
		
	else:
		err=os.system(f'mkdir -p {os.path.join(project_dir_path,"config")} {os.path.join(project_dir_path,"output")}')
		assert err==0,f'creating project directory resulted in error:{err}'
		err=os.system(f'{copyCmd} {os.path.join(baseDir,"runner","user_config.json")} {os.path.join(project_dir_path,"config")}')
		assert err==0,f'Copying config resulted in error:{err}'
		err=os.system(f'{copyCmd} -r {os.path.join(baseDir,"user_federates")} {project_dir_path}')
		assert err==0,f'Copying user_federates resulted in error:{err}'
		err=os.system(f'{copyCmd} -r {os.path.join(baseDir,"user_interface")} {project_dir_path}')
		assert err==0,f'Copying user_interface resulted in error:{err}'


@main.command(name="build")
@click.option("-t","--tag", required=True, help="Tag to be applied during docker build")
@click.option("-p","--python_cmd", required=False, default=defaultConfig['python_cmd'] ,\
	help="Python command to use i.e. python or python3")
@click.option("--nocache", required=False, type=bool, default=False, help="apply --no-cache option")
@click.option("--podman", required=False, default=defaultConfig['podman'], help="Use podman instead of docker")
def build(tag,python_cmd,nocache,podman):
	"""Builds a new Docker/Podman Image"""
	if nocache:
		directive=f'{python_cmd} "{os.path.join(baseDir,"build.py")}" --nocache true -t {tag}'
	else:
		directive=f'{python_cmd} "{os.path.join(baseDir,"build.py")}" -t {tag}'
	if podman:
		directive+=' --podman true'
	err=os.system(directive)
	assert err==0,f'Build resulted in error:{err} for directive={directive}'


@main.command(name="stop")
@click.option("--podman", required=False, default=defaultConfig['podman'], help="Use podman instead of docker")
def stop(podman):
	"""Stops all running instances of singlecontainerapp_* containers"""
	containerEngine='podman' if podman else 'docker'

	thisFile=f'temp_{uuid.uuid4().hex}.txt'
	os.system(f'{containerEngine} ps --filter name=singlecontainerapp > {thisFile}')
	fpath=os.path.join(os.getcwd(),thisFile)
	f=open(fpath); data=f.read(); f.close()
	data=re.sub(r'[ ]{3,}',',',data)
	data=data.splitlines()
	col=data[0].split(',')
	ind=col.index('CONTAINER ID')
	os.remove(f'{thisFile}')

	ids=[]
	for thisLine in data[1::]:
		ids.append(thisLine.split(',')[ind])

	for entry in ids:
		os.system(f'{containerEngine} stop -t 0 {entry}')


@main.command(name="set_default")
@click.option("-p","--python_cmd", required=False, default=defaultConfig['python_cmd'],\
	help="Python command to use i.e. python or python3")
@click.option("--podman", required=False, default=defaultConfig['podman'], help="Use podman instead of docker")
@click.option("-t","--tag", required=False, default=defaultConfig['tag'], help="Tag to be applied during docker build")
@click.option("-r","--run_as_admin", required=False, default=defaultConfig['run_as_admin'],\
	help="Should docker be run as root",type=bool)
def set_default(python_cmd,podman,tag,run_as_admin):
	"""Set default settings"""
	defaultConfig=json.load(open(defaultConfigPath))
	defaultConfig['python_cmd']=python_cmd
	defaultConfig['podman']=podman
	defaultConfig['tag']=tag
	defaultConfig['run_as_admin']=run_as_admin
	json.dump(defaultConfig,open(defaultConfigPath,'w'),indent=3)


@main.command(name="get_default")
def get_default():
	"""Get default settings"""
	defaultConfig=json.load(open(defaultConfigPath))
	print(json.dumps(defaultConfig,indent=3))


@main.command(name="list_tags")
@click.option("--podman", required=False, default=defaultConfig['podman'], help="Use podman instead of docker")
def list_tags(podman):
	"""Lists available image tags"""
	containerEngine='podman' if podman else 'docker'
	name='singlecontainerapp'
	os.system(f'{containerEngine} images > temp.csv')
	f=open('temp.csv'); data=f.read(); f.close()
	os.remove('temp.csv')

	data=re.sub(r'[ ]{3,}',',',data)

	df = pd.read_csv(StringIO(data))
	df[df.REPOSITORY==name]
	availableTags=list(set(df[df.REPOSITORY==name].TAG))
	print([f'{name}:{entry}' for entry in availableTags])


@main.command(name="gui_start")
@click.option("-r","--run_as_admin", required=False, default=defaultConfig['run_as_admin'],\
	help="Should docker be run as root",type=bool)
@click.option("--podman", required=False, default=defaultConfig['podman'], help="Use podman instead of docker")
def gui_start(run_as_admin,podman):
	"""Starts GUI containers -- uiruntime and uiserver"""
	containerEngine='podman' if podman else 'docker'

	directive=''
	if run_as_admin:
		directive+='sudo '
	directive+=f'{containerEngine} run --rm --name='

	f_rtout=open(os.path.join(baseDir,'logs','uiruntime_out'),'w')
	f_rterr=open(os.path.join(baseDir,'logs','uiruntime_err'),'w')
	f_servout=open(os.path.join(baseDir,'logs','uiserver_out'),'w')
	f_serverr=open(os.path.join(baseDir,'logs','uiserver_err'),'w')
	proc1=subprocess.Popen(shlex.split(f'{directive}uiruntime --net=host openenergydatainitiative/uiruntime:latest'),shell=False,\
		stdout=f_rtout,stderr=f_rterr)
	proc2=subprocess.Popen(shlex.split(f'{directive}uiserver --net=host openenergydatainitiative/uiserver:latest'),shell=False,\
		stdout=f_servout,stderr=f_serverr)


@main.command(name="gui_stop")
@click.option("-r","--run_as_admin", required=False, default=defaultConfig['run_as_admin'],\
	help="Should docker be run as root",type=bool)
@click.option("--podman", required=False, default=defaultConfig['podman'], help="Use podman instead of docker")
def gui_stop(run_as_admin,podman):
	containerEngine='podman' if podman else 'docker'

	directive=''
	if run_as_admin:
		directive+='sudo '

	os.system(directive+f'{containerEngine} stop -t=0 uiruntime')
	os.system(directive+f'{containerEngine} stop -t=0 uiserver')


@main.command(name="gui_status")
@click.option("-r","--run_as_admin", required=False, default=defaultConfig['run_as_admin'],\
	help="Should docker be run as root",type=bool)
@click.option("--podman", required=False, default=defaultConfig['podman'], help="Use podman instead of docker")
def gui_status(run_as_admin,podman):
	containerEngine='podman' if podman else 'docker'
	directive=''
	if run_as_admin:
		directive+='sudo '
	os.system(directive+f'{containerEngine} ps --filter name=ui')


@main.command(name="gui_update")
@click.option("-r","--run_as_admin", required=False, default=defaultConfig['run_as_admin'],\
	help="Should docker be run as root",type=bool)
@click.option("--podman", required=False, default=defaultConfig['podman'], help="Use podman instead of docker")
def gui_update(run_as_admin,podman):
	containerEngine='podman' if podman else 'docker'
	directive=''
	if run_as_admin:
		directive+='sudo '
	os.system(directive+"docker pull openenergydatainitiative/uiruntime:latest")
	os.system(directive+"docker pull openenergydatainitiative/uiserver:latest")


@main.command(name="gui_list_images")
@click.option("-r","--run_as_admin", required=False, default=defaultConfig['run_as_admin'],\
	help="Should docker be run as root",type=bool)
@click.option("--podman", required=False, default=defaultConfig['podman'], help="Use podman instead of docker")
def gui_list_images(run_as_admin,podman):
	containerEngine='podman' if podman else 'docker'
	directive=''
	if run_as_admin:
		directive+='sudo '
	os.system(directive+f'{containerEngine} images "ui*"')


@main.command(name="gui_get_logs")
def gui_get_logs():
	"""Gets the current uiruntime and uiserver logs"""

	data='\nuiserver\n'+'='*len('uiserver')+'\n\n'
	for entry in ['uiserver_out','uiserver_err']:
		f=open(os.path.join(baseDir,'logs',entry),'r')
		thisFileData=f.read()
		if thisFileData:
			data+=thisFileData+'\n\n'
		f.close()

	data+='uiruntime\n'+'='*len('uiruntime')+'\n\n'
	for entry in ['uiruntime_out','uiruntime_err']:
		f=open(os.path.join(baseDir,'logs',entry),'r')
		thisFileData=f.read()
		if thisFileData:
			data+=thisFileData+'\n\n'
		f.close()

	print(data)




if __name__ == '__main__':
	main()


