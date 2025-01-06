import os
import setuptools
import site


data_files=[]
#relativePath='lib'+site.getsitepackages()[0].split('lib')[1] #this doesn't work with Conda environment
site_packages = site.getsitepackages()
relativePath = 'lib'+next((path for path in site_packages if "lib" in path), None).split('lib')[1] #avoid errors when getsitepackages gives multiple path


for thisFolder in ['build','docs','logs','output','runner','tmp','user_federates','user_interface']:
	for root,dirnames,fnames in os.walk(thisFolder):
		data_files.append((f'{os.path.join(relativePath,"oedisi_single_container",root)}',\
		[os.path.join(os.path.abspath(root),fname) for fname in fnames]))

data_files.append((f'{os.path.join(relativePath,"oedisi_single_container")}',['build.py']))

setuptools.setup(name='oedisi_single_container',
	author = 'Karthikeyan Balasubramaniam',
	author_email='kbalasubramaniam@anl.gov',
	packages=setuptools.find_packages(),
	data_files=data_files,
	version='0.3.0',
	install_requires=['click==8.1.7','pandas>2.0.0'],
	entry_points={
		"console_scripts":["oedisisc = oedisi_single_container.cli:main"]
	}
)
