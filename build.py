import os
import sys
import pdb
import argparse
import shutil
import json
import subprocess

json_file = r"C://Users//splathottam//Box Sync//GitHub//oedi-si-single-container//specification.json" #need a way to specify path to specification file

def read_specification_and_clone_repository(json_file:str,target_directory:str,application:str):
	# Read the JSON file
	with open(json_file, 'r') as file:
		data = json.load(file)

	if not os.path.exists(target_directory):
		print(f"Creating {target_directory} since it was not found...")
		os.mkdir(target_directory)

	os.chdir(target_directory) # Change directory to application folder

	# Extract URL and tag
	url = data[application]['url']
	tag = data[application]['tag']

	# Extract the repository name from the URL
	repository_name = url.split('/')[-1]
	repository_path = os.path.join(target_directory,repository_name)

	if os.path.exists(repository_path):
		print("Found existing repository....removing...")
		shutil.rmtree(repository_path)# Delete the cloned repository if it exists
	
	# Clone the repository with the specific tag into the target directory
	subprocess.run(['git', 'clone', '--depth', '1', '--branch', tag, url])

	# Print the current working directory to verify
	print("Current Directory:", os.getcwd())
	
	return repository_path,repository_name

import re

def modify_dockerfile_content(dockerfile_path, application_directory):
	file_pattern = re.compile(r'COPY\s+([^\s]+)\s+([^\s]+)')
	modified_lines = []

	with open(dockerfile_path, 'r') as file:
		dockerfile_content = file.read()

	for line in dockerfile_content.splitlines():
		match = file_pattern.match(line)
		if match:
			src, dest = match.groups()
			print(f"Pre-pending {application_directory} to source path {src}")
			new_src = f'{application_directory}/{src}'
			modified_line = f'COPY {new_src} {dest}'
			modified_lines.append(modified_line)
		else:
			modified_lines.append(line)

	return '\n'.join(modified_lines)

if __name__=="__main__":
	preferredBuildOrder=['pnnl_dsse','datapreprocessor','dopf_ornl']
	fix_white_space=lambda x:'"'+x+'"' if len(x.split(' '))>1 else x
	parser=argparse.ArgumentParser()
	parser.add_argument('-t','--tag',help='tag to be applied during docker build',required=True)
	parser.add_argument('--nocache',help='apply --no-cache option',type=bool, required=False, default=False)
	parser.add_argument("--podman", required=False, default=False, help="Use podman instead of docker")
	args=parser.parse_args()

	engine='podman' if args.podman else 'docker'

	baseDir=os.path.dirname(os.path.abspath(__file__))
	buildDir=os.path.join(baseDir,'build')
	noCache='--no-cache' if args.nocache else ''

	if 'win' in sys.platform:
		isWindows=True
	else:
		isWindows=False

	tmpDir=os.path.join(baseDir,'tmp')
	if not os.path.exists(tmpDir):
		os.system(f'mkdir {tmpDir}')
	if '.gitignore' in os.listdir(tmpDir):# ensure that this is the correct directory
		f=open(os.path.join(baseDir,'tmp','.gitignore'))
		tempData=f.read()
		f.close()

		shutil.rmtree(tmpDir)
		os.system(f'mkdir {fix_white_space(tmpDir)}')

		f=open(os.path.join(baseDir,'tmp','.gitignore'),'w')
		f.write(tempData)
		f.close()

	data=''
	thisFolder=os.path.join(buildDir,'oedisi')
	f=open(os.path.join(thisFolder,'Dockerfile'))
	data+=f.read()+'\n'
	f.close()
	
	copyStatements=''
	if 'copy_statements.txt' in os.listdir(thisFolder):
		f=open(os.path.join(thisFolder,'copy_statements.txt'))
		copyStatements+=f.read()+'\n'
		f.close()

	contents=set(os.listdir(thisFolder)).difference(['Dockerfile'])
	for thisItem in contents:
		shutil.copy(os.path.join(thisFolder,thisItem),tmpDir)

	dockerItems=list(set(os.listdir(buildDir)).difference(['oedisi','datapreprocessor','dopf_ornl']))
	#dockerItems.append('datapreprocessor')
	
	if not set(dockerItems).difference(preferredBuildOrder):
		dockerItems=preferredBuildOrder
	dockerItems = ["pnnl_dopf","dopf_ornl"] #Add applications tob e build here
	modify_application_dockerfile = True
	for entry in dockerItems:
		thisFolder=os.path.join(buildDir,entry)
		print(f"Cloning to:{thisFolder}")
		repositoryFolder,repositoryName = read_specification_and_clone_repository(json_file,target_directory=thisFolder,application=entry)

		print(f"Opening Dockerfile and reading build commands in {repositoryFolder}...")		
		
		if modify_application_dockerfile:
			modified_dockerfile_data = modify_dockerfile_content(os.path.join(repositoryFolder,'Dockerfile'), repositoryName)
			data+=modified_dockerfile_data+'\n' #Read Dockerfile and append			
		else:
			f=open(os.path.join(repositoryFolder,'Dockerfile'))
			data+=f.read()+'\n' #Read Dockerfile and append
			f.close()
		if 'copy_statements.txt' in os.listdir(repositoryFolder):
			f=open(os.path.join(thisFolder,'copy_statements.txt'))
			copyStatements+=f.read()+'\n'
			f.close()
	
	print("Creating single container Dockerfile...")
	f=open(os.path.join(tmpDir,'Dockerfile'),'w') #Open a Dockerfile
	if isWindows:
		data+='\nRUN apt install -y dos2unix'
		f.write(data+'\n'+copyStatements+'\nENTRYPOINT dos2unix /home/runtime/runner/run.sh && '+\
			'/home/runtime/runner/run.sh')
	else:
		f.write(data+'\n'+copyStatements+'\nENTRYPOINT /home/runtime/runner/run.sh') #Append contents from individual application Dockerfiles to new Dockerfile
	f.close()

	# build
	print(f"Start build process using Dockerfile in {tmpDir}...")
	os.system(f'cd {fix_white_space(tmpDir)} && {engine} build {noCache} -t {args.tag} .')
