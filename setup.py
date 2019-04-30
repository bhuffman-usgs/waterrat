from setuptools import setup

setup(
    name = 'WaterRat',
    version = '1.0.0',
	description = 'AUV data viewer.',
    author = 'Brad J. Huffman',
    author_email = 'bhuffman@usgs.gov',
	packages = ['waterrat'],
	url = 'https://code.usgs.gov/bhuffman/waterrat',
	zip_safe = False,
    long_description = open('README.md').read(),
    include_package_data = True,
    install_requires = [
        'plotly == 3.1.0',
        'numpy == 1.15.0',
        'scipy == 1.1.0',
        'pandas == 0.23.3',
        'pyproj == 1.9.6',
        'utm == 0.4.2',
        'matplotlib == 2.2.2',
        'DateTime == 4.2',
        'dash == 0.39.0',
        'dash-core-components == 0.44.0',
        'dash-html-components == 0.14.0',
        'dash-renderer == 0.20.0',
	'cefpython3 == 66.0'
    ]
)
