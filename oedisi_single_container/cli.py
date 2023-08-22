import os
import sys
import uuid
import pdb

import click


baseDir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if 'win32' in sys.platform or 'cygwin' in sys.platform:
	isWindows=True
	copyCmd='robocopy'
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
@click.option("-t","--tag", required=False, default='singlecontainerapp:0.2.1', help="Should docker be run as root")
def run(project_dir_path,config,run_as_admin,tag):
	project_dir_path=os.path.abspath(project_dir_path)
	config=os.path.abspath(config)

	directive=''
	if run_as_admin:
		directive+='sudo '
	if isWindows:
		windowsVolumeMount(baseDir,project_dir_path,config,tag)
		name=uuid.uuid4().hex
		directive=f'docker run --name {name} '+\
			f'-v oedisisc_runtime:/home/runtime {tag}'
		os.system(directive)
		# copy output
		os.system(f'docker cp {name}:/home/output {os.path.join(project_dir_path)}')
		os.system(f'docker rm {name}')
	else:
		directive+=f'docker run --rm -v {os.path.join(baseDir,"runner")}:/home/runtime/runner '+\
			f'-v {config}:/home/runtime/runner/user_config.json '+\
			f'-v {os.path.join(project_dir_path,"user_federates")}:/home/runtime/user_federates '+\
			f'-v {os.path.join(baseDir,"user_interface")}:/home/runtime/user_interface '+\
			f'-v {os.path.join(project_dir_path,"output")}:/home/output {tag}'
		os.system(directive)

def windowsVolumeMount(baseDir,project_dir_path,configPath,tag):
	# check for missing volumes
	missingVolumes=[]
	for entry in ['oedisisc_runtime']:
		err=os.system(f'docker volume inspect {entry}')
		if err!=0:
			missingVolumes.append(entry)

	for entry in missingVolumes:
		err=os.system(f'docker volume create {entry}')

	# mount
	err=os.system(f'docker container create --name oedisisc_dummy -v oedisisc_runtime:/home/runtime {tag}')

	# copy
	os.system(f'robocopy {os.path.join(baseDir,"runner")} {os.path.join(project_dir_path,"runner")} *.*')
	os.system(f'robocopy {os.path.dirname(configPath)} {os.path.join(project_dir_path,"runner")} '+\
		configPath.split("\\")[-1])
	err=os.system(f'docker cp {os.path.join(project_dir_path,"runner")} oedisisc_dummy:/home/runtime')
	err=os.system(f'docker cp {os.path.join(baseDir,"user_interface")} oedisisc_dummy:/home/runtime')
	err=os.system(f'docker cp {os.path.join(project_dir_path,"user_federates")} oedisisc_dummy:/home/runtime')

	# delete
	err=os.system('docker rm oedisisc_dummy')

@main.command(name="init")
@click.option("-p","--project_dir_path", required=True, help="Path to template folder")
def init(project_dir_path):
	if isWindows:
		err=os.system(f'mkdir {os.path.join(project_dir_path,"config")} {os.path.join(project_dir_path,"output")}')
		err=os.system(f'{copyCmd} {os.path.join(baseDir,"runner")} {os.path.join(project_dir_path,"config")}')
		err=os.system(f'{copyCmd} {os.path.join(baseDir,"user_federates")} {os.path.join(project_dir_path,"user_federates")} /MIR')
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
@click.option("-p","--python_cmd", required=False, default='python3' ,help="Python command to use i.e. python or python3")
@click.option("--nocache", required=False, type=bool, default=False, help="apply --no-cache option")
def build(tag,python_cmd,nocache):
	if nocache:
		err=os.system(f'{python_cmd} {os.path.join(baseDir,"build.py")} --nocache true -t {tag}')
	else:
		err=os.system(f'{python_cmd} {os.path.join(baseDir,"build.py")} -t {tag}')
	assert err==0,f'Build resulted in error:{err} for directive={python_cmd} {os.path.join(baseDir,"build.py")} -t {tag}'


if __name__ == '__main__':
	main()
