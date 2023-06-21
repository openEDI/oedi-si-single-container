import setuptools

setuptools.setup(name='oedisi_single_container',
	author = 'Karthikeyan Balasubramaniam',
	author_email='kbalasubramaniam@anl.gov',
	include_package_data=False,
	version='0.2.0',
	install_requires=['click'],
	entry_points={
		"console_scripts":["oedisisc = oedisi_single_container.cli:main"]
	}
)
