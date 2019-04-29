Waterbody Rapid Assessment Tool (WaterRAT) 1.0
===============

WaterRAT is a Python application designed for 3D visualizations of data collected by [autonomous underwater vehicles (AUVs)](https://www.usgs.gov/centers/oki-water/science/autonomous-underwater-vehicles-auv?qt-science_center_objects=0#qt-science_center_objects) operated by the [United States Geological Survey](https://www.usgs.gov/). The application, distributed as a Python package, is run at the command line and is viewed in a compatible web browser under the address "localhost:8050" or "127.0.0.1:8050". This repository contains the structure to form a packaged version of the application including a configuration file necessary in its operation. 

Version Info: This package is still in a development state with potentially frequent modifcations, please check the release notes.

The application was developed in Python 3.6 on a Windows 10 Operating System and is a work in progress. The package is being released as preliminary software; please see the Disclaimer section for complete details. Currently, the application functions for a study area on the Little Back River in Savannah, GA. See "Initial Parameters" for information on how to change the location.

## How to Install Python 3.6 (and up)

1. [Download Miniconda (Windows 64-bit Miniconda installer for Python 3.x)](https://conda.io/en/latest/miniconda.html).
2. When the download is complete, run the Miniconda.exe installer. 
3. During the installation, when you are at the "Advanced Installation Options" window, check the box next to "Add Anaconda to my PATH environment variable." 

## How to Install Git

1. [Download Git (Windows 64-bit Git installer)](https://git-scm.com/download/win).
2. When the download is complete, run the Git.exe installer. 
3. During the installation, when you are at the "Adjusting your PATH environment" window, select the "Git from the command line and also..." option.

## How to Install the WaterRAT Package

1. Press the Win + R keys, type "cmd" (without quotations) in the box, and press "OK." The command line will open.
2. Type "conda create -n vepy python" (without quotations).  This will create a virtual environment named "vepy" to install the WaterRAT package to.
3. When the virtual environment is done being created, type "conda activate vepy" (without quotations).
4. Type "pip install git+https://github.com/bhuffman-usgs/waterrat" (without quotations).  The WaterRAT package will begin installing, including all its dependency packages.
5. 

## How to Start the Web Application

1. Open Windows File Explorer and navigate to the folder containing the config.ini configuration file.
2. Go to the "File" menu and select "Open command prompt" or "Open Windows PowerShell."
3. Type "python" (without quotation marks) into the command prompt and press enter. If you see ">>>" in the command prompt, the Python interperter is now running and you may proceed.
4. Type "import waterrat.suite as wrs" (without quotations) and press enter.
5. Type "wrs.appRun()" (without quotations) and press enter. The application will start.
6. A window will pop up and you will be prompted to select a data file. Select the AUV data file (CSV file type).
7. Open a web browser (Google Chrome is recommended) and go to the following website: localhost:8050

## Initial Parameters

#### The following parameters can be edited in the config.ini file:

Parameters related to the thalweg:

* **thal_lat**: Site-specific list of latitude coordinates for the center-line thalweg of the 3D models
* **thal_lon**: Site-specific list of longitude coordinates for the center-line thalweg of the 3D models

Parameters related to water column landmarks (landmarks that extend from the river bed to the water surface):

* **lm1_lat**: Site-specific list of latitude coordinates for landmarks that extend from the river bed to the water surface
* **lm1_lon**: Site-specific list of longitude coordinates for landmarks that extend from the river bed to the water surface
* **lm1_text**: Site-specific list of text labels for the bottom and top of landmarks that extend from the river bed to the water surface

Parameters related to bottom landmarks (landmarks that lie on the water surface):

* **lm2_lat**: Site-specific list of latitude coordinates for landmarks that lie on the water surface
* **lm2_lon**: Site-specific list of longitude coordinates for landmarks that lie on the water surface
* **lm2_text**: Site-specific list of text labels for landmarks that lie on the water surface

Parameters related to the Mapbox figure (map of study site):

* **map_zoom**: Site-specific zoom level of Mapbox figure (map of study site)
* **map_bearing**: Site-specific direction of Mapbox figure (map of study site); the direction you're facing, as measured from a compass
* **map_pitch**: Site-specific pitch of Mapbox figure in degrees

Parameters related to site width:

* **riv_w**: Site width

Parameters related to spacing and scale in the figures:

* **dx**: Grid spacing in the x direction for the 3D plot
* **dy**: Grid spacing in the y direction for the 3D plot
* **dz**: Grid spacing in the z direction for the 3D plot
* **asp_z**: Vertical scale for the 3D plot

## Troubleshooting

If you encounter errors, please review the following notes.

*  Git must be installed to install WaterRAT. Please follow the above directions ("How to Install Git").
*  WaterRAT will not work with Python version 2.x. Follow the above directions to install Python 3.x ("How to Install Python") and be sure to select the "Add Anaconda to my PATH environment variable" option during installation. 
*  When starting the application, you must run the command prompt from the same folder where the config.ini folder is located. 
*  The WaterRAT application currently works only for the Savannah River site. To view other sites the thalweg lat/longs will need to be edited in the config.ini file (see "Parameters related to the thalweg" above).
*  The AUV data input file should be a comma-seperated file.

## Security

WaterRAT runs locally on your computer and does not interact with the web. Upon running the program, WaterRAT will prompt you to upload a data file. Please be aware that WaterRAT will read this file and write an analyzed output file to the same location. WaterRAT will not use any sensitive information, such as personally identifiable information, internal system file paths, host names, or IP addresses. The WaterRAT application is not connected to any USGS networks. 

## Disclaimer

This software is preliminary or provisional and is subject to revision. It is being provided to meet the need for timely best science. The software has not received final approval by the U.S. Geological Survey (USGS). No warranty, expressed or implied, is made by the USGS or the U.S. Government as to the functionality of the software and related material nor shall the fact of release constitute any such warranty. The software is provided on the condition that neither the USGS nor the U.S. Government shall be held liable for any damages resulting from the authorized or unauthorized use of the software.
