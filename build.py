import os
import sys
import argparse


if __name__=="__main__":

	parser=argparse.ArgumentParser()
	parser.add_argument('-t','--tag',help='tag to be applied during docker build',required=True)
	args=parser.parse_args()

	baseDir=os.path.dirname(os.path.abspath(__file__))
	buildDir=os.path.join(baseDir,'build')

	tmpDir=os.path.join(baseDir,'tmp')
	if not os.path.exists(tmpDir):
		os.system(f'mkdir {tmpDir}')

	copyCmd='copy' if 'win' in sys.platform else 'cp'

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
	directive=f"{copyCmd} {' '.join(contents)} {tmpDir}"
	flag=os.system(directive)
	assert flag==0,f"directive: {directive} returned non-zero flag {flag}"
	os.chdir(buildDir)

	for entry in set(os.listdir(buildDir)).difference(['oedisi']):
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
			directive=f"{copyCmd} {' '.join(contents)} {tmpDir}"
			print(directive)
			flag=os.system(directive)
			assert flag==0,f"directive: {directive} returned non-zero flag {flag}"
			os.chdir(buildDir)

	os.chdir(baseDir)
	f=open(os.path.join(tmpDir,'Dockerfile'),'w')
	f.write(data+'\n'+copyStatements+'\nENTRYPOINT /home/runtime/runner/run.sh')
	f.close()

	# build
	os.system(f'cd {tmpDir} && docker build -t {args.tag} .')


