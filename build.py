import os
import sys
import pdb
import argparse
import shutil


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

	dockerItems=list(set(os.listdir(buildDir)).difference(['oedisi','datapreprocessor']))
	dockerItems.append('datapreprocessor')
	####
	if not set(dockerItems).difference(preferredBuildOrder):
		dockerItems=preferredBuildOrder
	for entry in dockerItems:
		thisFolder=os.path.join(buildDir,entry)
		f=open(os.path.join(buildDir,entry,'Dockerfile'))
		data+=f.read()+'\n'
		f.close()

		if 'copy_statements.txt' in os.listdir(thisFolder):
			f=open(os.path.join(thisFolder,'copy_statements.txt'))
			copyStatements+=f.read()+'\n'
			f.close()

		contents=set(os.listdir(os.path.join(buildDir,entry))).difference(\
			['Dockerfile','copy_statements.txt'])
		if contents:
			for thisItem in contents:
				print(os.path.join(thisFolder,thisItem),os.path.join(tmpDir,thisItem))
				shutil.copytree(os.path.join(thisFolder,thisItem),\
					os.path.join(tmpDir,thisItem),dirs_exist_ok=True)

	f=open(os.path.join(tmpDir,'Dockerfile'),'w')
	if isWindows:
		data+='\nRUN apt install -y dos2unix'
		f.write(data+'\n'+copyStatements+'\nENTRYPOINT dos2unix /home/runtime/runner/run.sh && '+\
			'/home/runtime/runner/run.sh')
	else:
		f.write(data+'\n'+copyStatements+'\nENTRYPOINT /home/runtime/runner/run.sh')
	f.close()

	# build
	os.system(f'cd {fix_white_space(tmpDir)} && {engine} build {noCache} -t {args.tag} .')


