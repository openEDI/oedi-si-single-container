import os
import sys
import pdb
import argparse


if __name__=="__main__":

	parser=argparse.ArgumentParser()
	parser.add_argument('-t','--tag',help='tag to be applied during docker build',required=True)
	parser.add_argument('--nocache',help='apply --no-cache option',type=bool, required=False, default=False)
	args=parser.parse_args()

	baseDir=os.path.dirname(os.path.abspath(__file__))
	buildDir=os.path.join(baseDir,'build')
	noCache='--no-cache' if args.nocache else ''

	if 'win' in sys.platform:
		isWindows=True
		copyCmd='robocopy'
	else:
		isWindows=False
		copyCmd='cp'

	tmpDir=os.path.join(baseDir,'tmp')
	if not os.path.exists(tmpDir):
		os.system(f'mkdir {tmpDir}')
	if '.gitignore' in os.listdir(tmpDir):# ensure that this is the correct directory
		f=open(os.path.join(tmpDir,'.gitignore'))
		tempData=f.read()
		f.close()
		if isWindows:
			os.system(f'rmdir /s {tmpDir} && mkdir {tmpDir}')
		else:
			os.system(f'rm -r {tmpDir} && mkdir {tmpDir}')
		f=open(os.path.join(tmpDir,'.gitignore'),'w')
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
	os.chdir(thisFolder)
	if isWindows:
		directive=f"{copyCmd} . {tmpDir} {' '.join(contents)}"
		flag=os.system(directive)
	else:
		directive=f"{copyCmd} {' '.join(contents)} {tmpDir}"
		flag=os.system(directive)
		assert flag==0,f"directive: {directive} returned non-zero flag {flag}"
	os.chdir(buildDir)

	dockerItems=list(set(os.listdir(buildDir)).difference(['oedisi','datapreprocessor']))
	dockerItems.append('datapreprocessor')
	for entry in dockerItems:
		thisFolder=os.path.join(buildDir,entry)
		f=open(os.path.join(buildDir,entry,'Dockerfile'))
		data+=f.read()+'\n'
		f.close()

		if 'copy_statements.txt' in os.listdir(thisFolder):
			f=open(os.path.join(thisFolder,'copy_statements.txt'))
			copyStatements+=f.read()+'\n'
			f.close()

		contents=set(os.listdir(os.path.join(buildDir,entry))).difference(['Dockerfile','copy_statements.txt'])
		if contents:
			os.chdir(thisFolder)
			if isWindows:
				directive=f"{copyCmd} . {tmpDir} {' '.join(contents)}"
			else:
				directive=f"{copyCmd} -r {' '.join(contents)} {tmpDir}"

			flag=os.system(directive)
			assert flag==0,f"directive: {directive} returned non-zero flag {flag}"
			os.chdir(buildDir)

	os.chdir(baseDir)
	f=open(os.path.join(tmpDir,'Dockerfile'),'w')
	if isWindows:
		data+='\nRUN apt install -y dos2unix'
		f.write(data+'\n'+copyStatements+'\nENTRYPOINT dos2unix /home/runtime/runner/run.sh && '+\
			'/home/runtime/runner/run.sh')
	else:
		f.write(data+'\n'+copyStatements+'\nENTRYPOINT /home/runtime/runner/run.sh')
	f.close()

	# build
	os.system(f'cd {tmpDir} && docker build {noCache} -t {args.tag} .')


