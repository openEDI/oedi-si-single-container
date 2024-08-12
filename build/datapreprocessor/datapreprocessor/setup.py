import setuptools

setuptools.setup(name='datapreprocessor',
	author = 'OEDISI ANL Team',
	author_email='kbalasubramaniam@anl.gov',
	include_package_data=False,
	version='0.1.0',
	install_requires=["tqdm>=4.64.0","py7zr>=0.19.0","tensorflow>=2.16.1","scikit-learn>=1.1.2","keras>=3.4.1"]
)
