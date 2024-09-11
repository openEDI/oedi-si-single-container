import os
import sys
import pdb
import argparse
import shutil
import json
import subprocess

def read_specification_and_clone_repository(specification_dict:dict,target_directory:str,repository_name:str="",show_details:bool=True,install:bool=False):
		
	if not os.path.exists(target_directory):
		print(f"Creating {target_directory} since it was not found...")
		os.mkdir(target_directory)

	os.chdir(target_directory) # Change directory to folder containing cloned repositories

	# Extract URL and tag/branch
	url = specification_dict['url']
	if 'tag' in specification_dict:
		tag = specification_dict['tag']
	elif 'branch' in specification_dict:
		tag = specification_dict['branch']
	else:
		raise ValueError("Expected either tag or branch for repository!")
	
	if 'dockerfilename' in specification_dict:
		dockerfilename = specification_dict['dockerfilename']
	else:
		dockerfilename = "Dockerfile"

	# Extract the repository name from the URL or utilize user supplied name
	if not repository_name:
		repository_name = f"{url.split('/')[-1]}"
	repository_path = os.path.join(target_directory,repository_name)
	print(f"Cloning to folder:{repository_name}")
	if os.path.exists(repository_path):
		print("Found existing repository....removing...")
		shutil.rmtree(repository_path)# Delete the cloned repository if it exists
	
	# Clone the repository with the specific tag into the target directory
	if show_details:
		subprocess.run(['git', 'clone', '--depth', '1', '--branch', tag, url,repository_path])
	else:
		subprocess.run(['git', 'clone', '--depth', '1', '--branch', tag, url,repository_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) #edirect the standard output and standard error to subprocess.DEVNULL
	
	if install:
		os.chdir(repository_name) # Change directory to cloned repository
		subprocess.run(['pip', 'install', '-e', '.'])
	
	return repository_name,dockerfilename

import re

#def modify_application_dockerfile_content(dockerfile_path:str, application_directory:str,work_directory:str):
def modify_application_dockerfile_content(modified_dockerfile_lines, application_directory:str,work_directory:str):
	file_pattern_copy = re.compile(r'COPY\s+([^\s]+)\s+([^\s]+)')
	file_pattern_workdir = r'^WORKDIR\s+(/.*)'
	modified_lines = []
	#print("Modifying Dockerfile...")
	#with open(dockerfile_path, 'r') as file:
	#	dockerfile_content = file.read()

	#for line in dockerfile_content.splitlines():
	for line in modified_dockerfile_lines:
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

	#return '\n'.join(modified_lines)
	return modified_lines

def remove_redundant_from_statements(dockerfile_path:str):
	print("Checking and removing redundant FROM statements...")
	
	with open(dockerfile_path, 'r') as file: # Read the Dockerfile
		lines = file.readlines()

	cleaned_lines = []
	from_found = False
	for line in lines:		
		if line.strip().startswith('FROM'): # Check if the line is a FROM statement
			if from_found:				
				print(f"Removing line:{line}") # Skip subsequent FROM statements
				continue
			from_found = True
		cleaned_lines.append(line)
	
	with open(dockerfile_path, 'w') as file: # Write the cleaned lines back to the Dockerfile
		file.writelines(cleaned_lines)

def update_oedisi_dockerfile(dockerfile_content:str, specification_config:dict):
	
	for repo_name, details in specification_config.items(): # Loop through each repository in the dictionary
		# Extract the repository URL and tag
		repo_url = details['url']
		repo_tag = details['tag']
		
		# Find the relevant git clone command in the Dockerfile content
		old_command = f"git clone --depth 1 --branch {repo_tag} {repo_url}.git"
		new_command = f"git clone --depth 1 --branch {repo_tag} {repo_url}.git"
		
		# Update the Dockerfile content
		dockerfile_content = dockerfile_content.replace(old_command, new_command)
		
	return dockerfile_content

def replace_home_paths(lines, replacement='/home/dsse_pnnl/'):
	
	pattern = r'/home' # Define the pattern to match '/home/'

	print(f"Replacing {pattern} with {replacement}")

	# Compile the regex pattern
	regex = re.compile(pattern)

	# Define the replacement pattern
	replacement_pattern = fr'\1{replacement}'

	# Process each line in the list
	modified_lines = [regex.sub(replacement, line) for line in lines]

	# Replace '/home/' with the specified replacement in each line
	#modified_lines = [regex.sub(replacement, line) for line in lines]

	return modified_lines

def replace_build_paths(lines, replacement='/home/dsse_pnnl/build'):	
	pattern = r'(LD_LIBRARY_PATH=/)?(\/)?build' # Define the pattern to match 'build', '/build', and 'LD_LIBRARY_PATH=/build'

	print(f"Replacing {pattern} with {replacement}")

	# Compile the regex pattern
	regex = re.compile(pattern)

	# Define the replacement pattern
	#replacement_pattern = fr'\1{replacement}'

	# Process each line in the list
	modified_lines = [regex.sub(replacement, line) for line in lines]

	return modified_lines
    
	

if __name__=="__main__":	
	dockerItems = ["datapreprocessor","dsse_pnnl","dopf_pnnl","dopf_ornl","ditto"] #Add applications to be included in the single container image here "datapreprocessor","dopf_ornl","dopf_pnnl"
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
	tmpFolder= "tmp"
	tmpDir=os.path.join(baseDir,tmpFolder)
	if not os.path.exists(tmpDir):
		os.system(f'mkdir {tmpDir}')
	if '.gitignore' in os.listdir(tmpDir):# ensure that this is the correct directory
		f=open(os.path.join(baseDir,tmpFolder,'.gitignore'))
		tempData=f.read()
		f.close()

		shutil.rmtree(tmpDir)
		os.system(f'mkdir {fix_white_space(tmpDir)}')

		f=open(os.path.join(baseDir,tmpFolder,'.gitignore'),'w')
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
	
	workDir = "/home"
	modify_application_dockerfile = True
	for applicationName in dockerItems:
		thisFolder=os.path.join(tmpDir)	#Clone application repository into tmp folder
		print(f"Copying/cloning {applicationName} to:{thisFolder}")		
		if applicationName in ["datapreprocessor","ditto"]: #If repository name in datapreprocess or utility			
			repositoryName = applicationName
			repositoryFolder = os.path.join(thisFolder,applicationName)
			dockerFileName = "Dockerfile"
			if os.path.exists(repositoryFolder):
				shutil.rmtree(repositoryFolder)
			shutil.copytree(os.path.join(buildDir,applicationName), repositoryFolder) # Copy the folder			
		else:			
			repositoryName,dockerFileName = read_specification_and_clone_repository(specification_dict["application"][applicationName],target_directory=thisFolder,repository_name=applicationName,show_details=args.showdetails)
			repositoryFolder = os.path.join(thisFolder,repositoryName)			
		
		dockerFilePath = os.path.join(repositoryFolder,dockerFileName)
		if modify_application_dockerfile:
			print(f"Opening, reading, and modifying build commands in Dockerfile:{dockerFilePath}...")
			with open(dockerFilePath, 'r') as file: # Read the Dockerfile
				dockerfile_lines = file.readlines()	
			#modified_dockerfile_data = modify_application_dockerfile_content(os.path.join(repositoryFolder,dockerFileName), applicationName,workDir)
			modified_dockerfile_lines = modify_application_dockerfile_content(dockerfile_lines, applicationName,workDir)

			if applicationName == "dsse_pnnl":				
				#prepend_path = f'/home/{applicationName}'
				#patterns = ['/home','/build']  # Patterns to prepend the path to
				#modified_dockerfile_lines = prepend_paths(modified_dockerfile_lines, applicationName, patterns)
				modified_dockerfile_lines = replace_home_paths(modified_dockerfile_lines, replacement=f'/home/{applicationName}')
				modified_dockerfile_lines = replace_build_paths(modified_dockerfile_lines, replacement=f'/home/{applicationName}/build')
				modified_dockerfile_lines.append(f'RUN chmod +x /home/{applicationName}/ekf_federate/state-estimator-gadal') #permission need to be changed for this executable for dsse_pnnl to work

			modified_dockerfile_data = '\n'.join(modified_dockerfile_lines) #Join list of strings to one file
			data+= '\n' + f'#{applicationName}' +'\n' + modified_dockerfile_data +'\n' #Read Dockerfile and append
			
		else:
			print(f"Opening and reading build commands in Dockerfile:{dockerFilePath}...")
			f=open(dockerFilePath)
			data+=f.read()+'\n' #Read Dockerfile and append
			f.close()
		if 'copy_statements.txt' in os.listdir(repositoryFolder):
			f=open(os.path.join(thisFolder,'copy_statements.txt'))
			copyStatements+=f.read()+'\n'
			f.close()
	
	print("Creating single container Dockerfile...")
	f=open(os.path.join(tmpDir,'Dockerfile'),'w') #Open a Dockerfile
	if isWindows:
		data+='\nRUN apt-get update && apt-get install -y dos2unix'
		data+='\nRUN mkdir -p /home/outputs' #create outputs folder
		f.write(data+'\n'+copyStatements+'\nENTRYPOINT dos2unix /home/runtime/runner/run.sh && '+\
			'/home/runtime/runner/run.sh')
	else:
		f.write(data+'\n'+copyStatements+'\nENTRYPOINT /home/runtime/runner/run.sh') #Append contents from individual application Dockerfiles to new Dockerfile
	f.close()

	remove_redundant_from_statements(os.path.join(tmpDir,'Dockerfile'))	#remove additional from statements

	# build
	print(f"Start build process using Dockerfile in {tmpDir}...")
	os.system(f'cd {fix_white_space(tmpDir)} && {engine} build {noCache} -t {args.tag} .')
