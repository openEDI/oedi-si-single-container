import os
import shutil
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

def modify_application_dockerfile_content(dockerfile_path:str, application_directory:str,work_directory:str):
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
			
			elif '/simulation' in line:
				print(f"Found '/simulation' in line {line.strip()}")			
				line = line.replace('/simulation', f'{work_directory}/{application_directory}')
				modified_lines.append(line)

			else:
				modified_lines.append(line)
		else:
			modified_lines.append(line)

	return '\n'.join(modified_lines)

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

def modify_dockerfile_cp_cd(dockerfile_path, append_path):
	# Compile regular expressions for search patterns
	patterns = [re.compile(pattern) for pattern in [r'^cp /build', r'^cd /build']]
	
	with open(dockerfile_path, 'r') as file: # Read the Dockerfile content
		lines = file.readlines()

	# Iterate over each line and modify it if necessary
	modified_lines = []
	for line in lines:
		for pattern in patterns:
			if pattern.search(line):				
				line = f"{line.strip()} {append_path}\n" # Append the append_path to the matched line
				break  # Exit the inner loop if a match is found
		modified_lines.append(line)
	
	with open(dockerfile_path, 'w') as file: # Write the modified content back to the Dockerfile
		file.writelines(modified_lines)
