import os
import sys
import pdb
import argparse
import shutil
import json
import subprocess

def read_specification_and_clone_repository(specification_dict:dict,target_directory:str,application:str,show_details:bool=True,install:bool=False):
		
	if not os.path.exists(target_directory):
		print(f"Creating {target_directory} since it was not found...")
		os.mkdir(target_directory)

	os.chdir(target_directory) # Change directory to folder containing cloned repositories

	# Extract URL and tag
	url = specification_dict[application]['url']
	tag = specification_dict[application]['tag']

	# Extract the repository name from the URL
	repository_name = url.split('/')[-1]
	repository_path = os.path.join(target_directory,repository_name)

	if os.path.exists(repository_path):
		print("Found existing repository....removing...")
		shutil.rmtree(repository_path)# Delete the cloned repository if it exists
	
	# Clone the repository with the specific tag into the target directory
	if show_details:
		subprocess.run(['git', 'clone', '--depth', '1', '--branch', tag, url])
	else:
		subprocess.run(['git', 'clone', '--depth', '1', '--branch', tag, url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) #edirect the standard output and standard error to subprocess.DEVNULL

	# Print the current working directory to verify
	print("Current Directory:", os.getcwd())

	if install:
		os.chdir(repository_name) # Change directory to cloned repository
		subprocess.run(['pip', 'install', '-e', '.'])
	
	return repository_path,repository_name

import re

def modify_application_dockerfile_content(dockerfile_path, application_directory,work_directory):
	file_pattern_copy = re.compile(r'COPY\s+([^\s]+)\s+([^\s]+)')
	file_pattern_workdir = r'^WORKDIR\s+(/.*)'
	modified_lines = []
	print("Modifying Dockerfile...")
	with open(dockerfile_path, 'r') as file:
		dockerfile_content = file.read()

	for line in dockerfile_content.splitlines():
		match_copy = file_pattern_copy.match(line)
		match_workdir = re.match(file_pattern_workdir, line.strip())
		if not application_directory == "datapreprocessor":
			if match_copy:
				src, dest = match_copy.groups()
				print(f"Pre-pending {application_directory} to source path {src}")
				new_src = f'{application_directory}/{src}'
				new_dest = f'{application_directory}/{dest}'
				modified_line = f'COPY {new_src} {dest}'
				modified_lines.append(modified_line)		
					
			elif match_workdir:
				existing_workdir = match_workdir.group(1) # Extract the existing directory from the regex match
				new_work_directory = f'{work_directory}/{application_directory}'
				print(f"Changing work directory to {new_work_directory}")
				modified_lines.append(f'WORKDIR {new_work_directory}') # Replace the line with the new WORKDIR directive
			
			else:
				modified_lines.append(line)
		else:
			modified_lines.append(line)

	return '\n'.join(modified_lines)

def remove_redundant_from_statements(dockerfile_path):
	print("Checking and removing redundant FROM statements...")
	# Read the Dockerfile
	with open(dockerfile_path, 'r') as file:
		lines = file.readlines()

	cleaned_lines = []
	from_found = False
	for line in lines:
		# Check if the line is a FROM statement
		if line.strip().startswith('FROM'):
			if from_found:
				# Skip subsequent FROM statements
				print(f"Removing line:{line}")
				continue
			from_found = True
		cleaned_lines.append(line)

	# Write the cleaned lines back to the Dockerfile
	with open(dockerfile_path, 'w') as file:
		file.writelines(cleaned_lines)

def update_oedisi_dockerfile(dockerfile_content, specification_config):
	# Loop through each repository in the dictionary
	for repo_name, details in specification_config.items():
		# Extract the repository URL and tag
		repo_url = details['url']
		repo_tag = details['tag']
		
		# Find the relevant git clone command in the Dockerfile content
		old_command = f"git clone --depth 1 --branch {repo_tag} {repo_url}.git"
		new_command = f"git clone --depth 1 --branch {repo_tag} {repo_url}.git"
		
		# Update the Dockerfile content
		dockerfile_content = dockerfile_content.replace(old_command, new_command)
		
	return dockerfile_content

if __name__=="__main__":
	preferredBuildOrder=['datapreprocessor','pnnl_dsse','dopf_ornl']
	fix_white_space=lambda x:'"'+x+'"' if len(x.split(' '))>1 else x
	parser=argparse.ArgumentParser()
	parser.add_argument('-t','--tag',help='tag to be applied during docker build',required=True)
	parser.add_argument('--nocache',help='apply --no-cache option',type=bool, required=False, default=False)
	parser.add_argument("--podman", required=False, default=False, help="Use podman instead of docker")
	parser.add_argument("--showdetails", required=False, default=False, help="Show more details")
	args=parser.parse_args()

	engine='podman' if args.podman else 'docker'

	baseDir=os.path.dirname(os.path.abspath(__file__))
	buildDir=os.path.join(baseDir,'build')
	noCache='--no-cache' if args.nocache else ''

	specification_file = os.path.join(baseDir,"specification.json") # #Specification file is in the base directory
	with open(specification_file, 'r') as file: # Read the JSON file
		specification_dict = json.load(file)
	
	if 'win' in sys.platform:
		isWindows=True
	else:
		isWindows=False

	tmpDir=os.path.join(baseDir,'tmp2')
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
	oedisi_dockerfile = f.read()
	oedisi_dockerfile = update_oedisi_dockerfile(oedisi_dockerfile, specification_dict["oedisi"])	
	data+=oedisi_dockerfile +'\n'
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
	dockerItems = ["datapreprocessor","pnnl_dopf","dopf_ornl"] #Add applications to be build here
	work_dir = "/home"
	modify_application_dockerfile = True
	for entry in dockerItems:
		thisFolder=os.path.join(tmpDir)	#Clone application repository into tmp folder
		print(f"Copying/cloning {entry} to:{thisFolder}")
		if entry == "datapreprocessor":	
			targetpath = os.path.join(thisFolder,"datapreprocessor")
			if os.path.exists(targetpath):
				shutil.rmtree(targetpath)
			shutil.copytree(os.path.join(buildDir,"datapreprocessor"), targetpath) # Copy the folder
			repositoryFolder = os.path.join(thisFolder,"datapreprocessor")
			repositoryName = "datapreprocessor"
		else:			
			repositoryFolder,repositoryName = read_specification_and_clone_repository(specification_dict["application"],target_directory=thisFolder,application=entry,show_details=args.showdetails)
			print(f"Opening Dockerfile and reading build commands in {repositoryFolder}...")		
			
		if modify_application_dockerfile:			
			modified_dockerfile_data = modify_application_dockerfile_content(os.path.join(repositoryFolder,'Dockerfile'), repositoryName,work_dir)
			data+= '\n' + f'#{repositoryName}' +'\n' + modified_dockerfile_data +'\n' #Read Dockerfile and append			
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

	remove_redundant_from_statements(os.path.join(tmpDir,'Dockerfile'))	#remove additional from statements

	# build
	print(f"Start build process using Dockerfile in {tmpDir}...")
	os.system(f'cd {fix_white_space(tmpDir)} && {engine} build {noCache} -t {args.tag} .')
