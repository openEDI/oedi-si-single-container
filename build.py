import os
import sys
import pdb
import argparse
import shutil
import json

from build_utilities import read_specification_and_clone_repository,modify_application_dockerfile_content,remove_redundant_from_statements,update_oedisi_dockerfile

if __name__=="__main__":	
	dockerItems = ["datapreprocessor","dsse_pnnl","dopf_pnnl","dsse_ornl","dopf_ornl","ditto"] #Add applications to be included in the single container image here "datapreprocessor","dopf_ornl","dopf_pnnl"
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
	tmpFolder= "tmp" #Specificy temporary folder which will be used while build the single container image
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
	
	work_dir = "/home"
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
		print(f"Opening Dockerfile and reading build commands in {repositoryFolder}...")		
			
		if modify_application_dockerfile:
			if applicationName == "dsse_pnnl":
				dsse_pnnl_folder=os.path.join(buildDir,'pnnl_dsse')
				f=open(os.path.join(dsse_pnnl_folder,'copy_statements.txt')) #Read dsse_pnnl from here since it is too hard to modify
				modified_dockerfile_data=f.read()+'\n'
				f.close()
				modified_dockerfile_data+= f'RUN chmod +x /home/{applicationName}/ekf_federate/state-estimator-gadal' +'\n' #permission need to be changed for this executable for dsse_pnnl to work
			else:
				modified_dockerfile_data = modify_application_dockerfile_content(os.path.join(repositoryFolder,dockerFileName), applicationName,work_dir)
			data+= '\n' + f'#{applicationName}' +'\n' + modified_dockerfile_data +'\n' #Read Dockerfile and append
			#if applicationName == "dsse_pnnl":
			#	data+= '\n' + f'RUN chmod +x /home/{applicationName}/ekf_federate/state-estimator-gadal' +'\n' #permission need to be changed for this executable for dsse_pnnl to work
		else:
			f=open(os.path.join(repositoryFolder,dockerFileName))
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
