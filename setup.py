import setuptools

setuptools.setup(name='oedisi_single_container',
	author = 'Karthikeyan Balasubramaniam',
	author_email='kbalasubramaniam@anl.gov',
	packages=setuptools.find_packages(),
	include_package_data=False,
	version='0.3.0',
	install_requires=['click==8.1.7','pandas==2.0.3'],
	entry_points={
		"console_scripts":["oedisisc = oedisi_single_container.cli:main"]
	}
)
