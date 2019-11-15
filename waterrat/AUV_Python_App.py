# Import all necessary libraries and functions
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colorbar import ColorbarBase
import os, sys, pyproj, utm, math, json, configparser
from .AUV_Python_Func import stringSubReplace, fDialog, case_load, voxelAlpha, voxelMat, euc_dist, equal_dist, find_div, fdd, fdun, curvey_grid, centroid_2D, generateColorbar, tickval2text, lay_2d, areaArbTri, mplToURI, indexByBin
import plotly.graph_objs as go
from scipy.interpolate import griddata as griddata
from scipy import interpolate as interp
import numpy as np
from numpy import square as ms, sqrt as msqrt
import pandas as pd
from dash import Dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input as dashIn, Output as dashOut, State as dashState
from dash.exceptions import PreventUpdate as dashNoUpdate
from textwrap import dedent

######################### Get File Directory for Processing #########################
# Get our current working directory
cwd = os.getcwd()

# Set the last directory text file filepath (storage of last used data file)
lastdir = cwd + '\\LastDirectory.txt'
# Check if the file exists ...
if os.path.isfile(lastdir):
    # Open the last directory file
    f = open(lastdir, 'r')
    # Get the number of lines in the file
    L_num = len(f.readlines())
    # Make sure the line number is equal to 1
    if L_num == 1:
        # Jump back to start of file
        f.seek(0)
        # Get line content
        fl = f.readline()
        if len(fl) == 0:
            startDir = '/'
        else:
            # Set the starting directory to what was in the last directory file
            startDir = fl
    # If the line number is not 1 ...
    else:
        # Set the starting directory to /
        startDir = '/'
    # Close file
    f.close()  

    # Clear variables from memory
    del f, L_num, fl
# If the file path does not exist ...
else:
    # Set the starting directory to /
    startDir = '/'
#####################################################################################

######################### Browse to Select Log File #########################
# Create the gui instance
fgui = fDialog('Load Processed', startDir, ('Comma Seperated File', '*.csv'), '_processed')
# Fire up the gui
fgui.master.mainloop()

# Extract the data file filepath from the gui
data_fname = fgui.filename
# Extract the parent directory for the data file
data_fpar = os.path.dirname(data_fname)
# Open the last directory text file
f = open(lastdir, 'w')
# Clear the file's contents
f.truncate()
# Save the data file's directory to last directory file
f.write('%s' % data_fpar)
# Close the file
f.close()

# Clear variables from memory
del fDialog, lastdir, startDir, fgui, data_fpar
##############################################################################

######################### Load Variables For 3D Viewer #########################
paramPath = cwd + '\\config.ini'
config = configparser.ConfigParser()
config.read(paramPath)

# Load the list variables
thal_lat = np.asarray(json.loads(config.get('viewerVariables', 'thal_lat')))
thal_lon = np.asarray(json.loads(config.get('viewerVariables', 'thal_lon')))
# PltVar = json.loads(config.get('viewerVariables', 'PltVar'))
PltVar = ["Temp", "SpCond", "Sal", "Dens", "pH", "Turb", "DO", "Chl", "BGA-PC", "BGA-PE", "Rhod"]

# Attempt to load the landmark list variables
try:
    lm1_lat = np.asarray(json.loads(config.get('viewerVariables', 'lm1_lat')))
    lm1_lon = np.asarray(json.loads(config.get('viewerVariables', 'lm1_lon')))
    lm1_text = json.loads(config.get('viewerVariables', 'lm1_text'))
    lmc1 = 1
except Exception as ex:
    if ex.__class__.__name__ == 'NoOptionError':
        lmc1 = 0
        pass
    elif ex.__class__.__name__ == 'JSONDecodeError':
        print('Error:  Use double quotes for text in landmark 1 text list!')
        sys.exit()
    else:
        print('Error:  Could not properly load landmark 1 info!')
        sys.exit()
try:
    lm2_lat = np.asarray(json.loads(config.get('viewerVariables', 'lm2_lat')))
    lm2_lon = np.asarray(json.loads(config.get('viewerVariables', 'lm2_lon')))
    lm2_text = json.loads(config.get('viewerVariables', 'lm2_text'))
    lmc2 = 1
except Exception as ex:
    if ex.__class__.__name__ == 'NoOptionError':
        lmc2 = 0
        pass
    elif ex.__class__.__name__ == 'JSONDecodeError':
        print('Error:  Use double quotes for text in landmark 2 text list!')
        sys.exit()
    else:
        print('Error:  Could not properly load landmark 2 info!')
        sys.exit()

# Load the integer variables
swid = 8 

# Load the float variables
riv_w = config.getfloat('viewerVariables', 'riv_w')  
dx = config.getfloat('viewerVariables', 'dx')  
dy = config.getfloat('viewerVariables', 'dy')   
dz = config.getfloat('viewerVariables', 'dz')   
# tube_size = config.getfloat('viewerVariables', 'tube_size')   
map_zoom = config.getfloat('viewerVariables', 'map_zoom')   
map_bearing = config.getfloat('viewerVariables', 'map_bearing')   
map_pitch = config.getfloat('viewerVariables', 'map_pitch')   
asp_z = config.getfloat('viewerVariables', 'asp_z')  

# 3d and 2d figure margin control
marg3d = 0
marg = 55
#######################################################################################

# Import the processed dataframe file
dat_df = pd.read_csv(data_fname)
# Clear the data filename from memory
del data_fname

# Assign data in df to variables
zone = utm.from_latlon(dat_df['DDLat'][0], dat_df['DDLong'][0])
zone = str(zone[2]) + zone[3]
# Generate a projection to convert lat/lon to easting/northing (meters) based on the UTM zone and the WGS84 ellipsoid
p = pyproj.Proj(proj = 'utm', zone = zone, ellps = 'WGS84')
# Apply the projection
x, y = p(dat_df['DDLong'].values, dat_df['DDLat'].values)

### Step 1: Filter data by specific waypoints ###
# Convert x and y (the projections in meters from lat and lon) to feet
x = x*3.28084
y = y*3.28084
# Grab the depth of the AUV (z) and the bed depth (d)
z = dat_df['DFS Depth (ft)'].values
d = dat_df['Total Water Column (ft)'].values
# Make the z and d arrays negative
z = -z
d = -d
# Truncate x, y, z and d using the filter indexes marking the starting and ending waypoints of interest
x = x
y = y 
z = z 
d = d 

### Step 2: Produce a curvilinear grid that follows along the thalwag of the site of interest ###
# Project thalwag lat/lons to the cartesian coordinate plane
thal_x, thal_y = p(thal_lon, thal_lat)
# Convert the projection to feet
thal_x = (thal_x*3.28084)
thal_y = (thal_y*3.28084)
# Remove the thalwag lat/lon coordiantes from memory
del thal_lon, thal_lat

# Check the first landmark control status
if lmc1 == 1:
    # Project the landmark lat/lons to cartesian coordinates
    lm1_x, lm1_y = p(lm1_lon, lm1_lat)
    # Convert the projection to feet
    lm1_x = (lm1_x*3.28084)
    lm1_y = (lm1_y*3.28084)
    # Remove the landmark lat/lon coordinates from memory
    del lm1_lon, lm1_lat
# Check the second landmark control status
if lmc2 == 1:
    # Project the landmark lat/lons to cartesian coordinates
    lm2_x, lm2_y = p(lm2_lon, lm2_lat)
    # Convert the projections to feet
    lm2_x = (lm2_x*3.28084)
    lm2_y = (lm2_y*3.28084)
    # Remove the landmark lat/lon coordinates from memory
    del lm2_lon, lm2_lat

# Get the min and max coordinates for the AUV track
xlow = min(x)
xhigh = max(x)
ylow = min(y)
yhigh = max(y)

# Loop over the thalwag and check if the points are within the AUV tracks bounding box
for i in range(0, len(thal_x)):
    # A = (xlow, ylow) , B = (xlow, yhigh) , C = (xhigh, yhigh) , D = (xhigh, ylow)
    # P = (tx, ty)
    tx = thal_x[i]
    ty = thal_y[i]
    # Calculate the sum of the areas of APD, DPC, CPB and PBA
    triSum = areaArbTri(xlow, ylow, tx, ty, xhigh, ylow) + areaArbTri(xhigh, ylow, tx, ty, xhigh, yhigh) + areaArbTri(xhigh, yhigh, tx, ty, xlow, yhigh) + areaArbTri(tx, ty, xlow, yhigh, xlow, ylow)
    # Calculate the area of ABCD or the sum of the areas of ABC and ACD
    rectSum = areaArbTri(xlow, ylow, xlow, yhigh, xhigh, yhigh) + areaArbTri(xlow, ylow, xhigh, yhigh, xhigh, ylow)

    # If the sum of the area of the triangles with point P is greater than the area of the rectangle 
    #   then P is outside the rectangle (So if the diff is less than 0.001 then P is inside)
    if ((triSum - rectSum) < 0.001):
        if (i != 0):
            # Use the previous points index
            iStart = i - 1
        else:
            iStart = 0
        # Exit the loop
        break
# Loop backwards over the thalwag and check if the points are within the AUV tracks bounding box
for i in range((len(thal_x) - 1), 0, -1):
    # A = (xlow, ylow) , B = (xlow, yhigh) , C = (xhigh, yhigh) , D = (xhigh, ylow)
    # P = (tx, ty)
    tx = thal_x[i]
    ty = thal_y[i]
    # Calculate the sum of the areas of APD, DPC, CPB and PBA
    triSum = areaArbTri(xlow, ylow, tx, ty, xhigh, ylow) + areaArbTri(xhigh, ylow, tx, ty, xhigh, yhigh) + areaArbTri(xhigh, yhigh, tx, ty, xlow, yhigh) + areaArbTri(tx, ty, xlow, yhigh, xlow, ylow)
    # Calculate the area of ABCD or the sum of the areas of ABC and ACD
    rectSum = areaArbTri(xlow, ylow, xlow, yhigh, xhigh, yhigh) + areaArbTri(xlow, ylow, xhigh, yhigh, xhigh, ylow)

    # If the sum of the area of the triangles with point P 
    #   is greater than the area of the rectangle then
    #   P is outside the rectangle
    if ((triSum - rectSum) < 0.001):
        if (i != (len(thal_x) - 1)):
            # Use next point
            iEnd = i + 1
        else:
            iEnd = i
        # Exit the loop
        break

while (iEnd - iStart) < 4:
    iStart -= 1
    iEnd += 1

# Use the iStart and iEnd indices to focus the model area
thal_x = thal_x[iStart:iEnd]
thal_y = thal_y[iStart:iEnd]
# Assign the base values for x and y
base_x = thal_x[0]
base_y = thal_y[0]
# Remove the lowest value from the easting and northing for legability
x = x - base_x
y = y - base_y
thal_x = thal_x - base_x
thal_y = thal_y - base_y
if lmc1 == 1:
    lm1_x = lm1_x - base_x
    lm1_y = lm1_y - base_y
if lmc2 == 1:
    lm2_x = lm2_x - base_x
    lm2_y = lm2_y - base_y

# Put a splined fit through the thalwag points (two-part spline from SciPy)
tck, u = interp.splprep([thal_x, thal_y], s=0)
ns = 100
spl_x, spl_y = interp.splev(np.linspace(0, 1, ns), tck)
# Get the euclidean distances from the spline points
dist = euc_dist(spl_x, spl_y)
# Find the largest and smallest jumps
min_d = np.min(dist[1:])
max_d = np.max(dist)
### Begin routine to match the distances between the spline points to the dx defined at the beginning ###
# Set a minimum distance between slipe points 
d_if = min((dx/4), 20)
# Is the largest distance between the splined points greater than (dx/4) or 20 ft (whichever is less)...
if max_d > d_if:
    # Set a loop boolean value to control entry and exit of the while loop
    loop = False
    # Add h number of points to break a linear space between 0 and 1 into to be fed into interp.splev
    h = 1
    # Start the while loop
    while loop == False:
        ### Implement a newton rhapson scheme and use a central difference O(h^2) derivative for ff which is an error function i.e. the error of the max distance between spline points to d_if
        # Using ns + h find the next max distance (forward step)
        # Get a spline from the tck variable computed earlier from interp.splprep
        ffspl_x, ffspl_y = interp.splev(np.linspace(0, 1, (ns + h)), tck)
        # Get the euclidean distances between the spline points
        ffdist = euc_dist(ffspl_x, ffspl_y)
        # Find the max point spacing
        ffmax = np.max(ffdist)
        # Find the min point spacing
        ffmin = np.min(ffdist[1:])

        # Using ns - h find the previous max distance (backward step)
        # Get a spline from the tck variable computed earlier from interp.splprep
        fbspl_x, fbspl_y = interp.splev(np.linspace(0, 1, (ns - h)), tck)
        # Get the euclidean distances between the spline points
        fbdist = euc_dist(fbspl_x, fbspl_y)
        # Find the max point spacing
        fbmax = np.max(fbdist)

        # Evaluate if the max and min distances (for the forward step) are close to dx/4...
        if (ffmax <= d_if):
            # If they are, exit the loop by setting loop equal to true
            loop = True
            # Use the forward step spline points
            spl_x = np.copy(ffspl_x)
            spl_y = np.copy(ffspl_y)
            # Get the euclidean distance from the forward step spline
            dist = np.copy(ffdist)

        # If not ...
        else:
            # ff = abs(maxdist - d_if) where maxdist is equivalent to a nest of functions relying on the variable ns --> np.max(euc_dist(interp.splev(np.linspace(0, 1, ns), tck)))
            # ff(ns)
            ff = abs(max_d - d_if)
            # ff'(ns) [central scheme]
            fdir = (abs(ffmax - d_if) - abs(fbmax - d_if))/(2*h)
            # Eval the next ns as nn and round off the decimal place
            nn = round(ns - (ff/fdir))
            # Get the step size
            h = abs(nn - ns)
            # Set a new ns value
            ns = nn
            # Evaluate the max_d for the new ns
            spl_x, spl_y = interp.splev(np.linspace(0, 1, ns), tck)
            # Get the euclidean distances from the spline points
            dist = euc_dist(spl_x, spl_y)
            # Find the max point spacing
            max_d = np.max(dist)

            del ff, fdir, nn
            
    del loop, h, ffspl_x, ffspl_y, ffdist, ffmax, ffmin, fbspl_x, fbspl_y, fbdist, fbmax

# Get the total distance along the curve
td = np.sum(dist)
td = (math.floor(td/dx)*dx)
# Extract at equal spaces along the curve
thal_x, thal_y, m_norm, b_norm = equal_dist(spl_x, spl_y, td, dx)
# Generate the distance axis along the thalwag for a grid axis
thal_d = euc_dist(thal_x, thal_y)
# Define a rounding function to apply to a numpy array
rounder = lambda t: (round(t/dx)*dx)
# Take the thalwag distance array, perform a cummulative sum on it, and round off the decimal place
xgrid_vec = np.array([rounder(x) for x in np.cumsum(thal_d)])
# Clear the rounder function and thalwag distance variable and remove other variables
del rounder, thal_d, tck, u, ns, spl_x, spl_y, dist, min_d, max_d, d_if, td
#########################################################################################################

### Create the x, y and z grid vectors ###
# Check divisions of river width and set to a division that yields no remainder that is closest to the original division
halfWidth = (riv_w/2)
if halfWidth % dy != 0:
    halfWidth = fdun(halfWidth, halfWidth, dy, 1.0)
    riv_w = halfWidth*2
# Set number of intervals for the lateral axis
ny = ((riv_w)/dy) + 1
while int(ny) < (swid - 1):
    dy = fdd(riv_w, dy, dy, 5)
    ny = ((riv_w)/dy) + 1
# Set the lateral grid points
ygrid_vec = np.linspace(-1*(riv_w/2), (riv_w/2), int(ny))
y_cen = ((ygrid_vec.shape[0] - 1)/2)

def discretize_for_slider(xrng, dx, inter, i):
    # Normalize the range by the intervals
    xrng = int(xrng/inter)
    # Get the dx value as an integer, normalized by the intervals
    dx = int(dx/inter)
    # If the dx value is less than the interval
    if dx == 0:
        dx = 1
    # Verify dx divides xrng with no remainder, if not find the nearest value that does
    if xrng % dx != 0:
        dx = find_div(xrng, dx, 1)
    # If the number of ticks is less than the number of slider ticks...
    while int((xrng/dx) + 1) < (i - 1):
        # Find the next smaller dx that divides xrng with no remainder
        dx = fdd(xrng, dx, dx, 1)
    # Return dx and the number of ticks
    return (dx*inter), int((xrng/dx) + 1)

# Get the deepest value in the array
zrng = math.ceil(-min(z))
# Define the intervals that dz can move in
inter = 0.1
dznew, nz = discretize_for_slider(zrng, dz, inter, swid)
if dznew != dz:
    print("A dz value of %.1f is not compatible for this dataset.  The value will be set to %.1f." % (dz, dznew))
    dz = dznew
# Set the vertical grid points
zgrid_vec = np.linspace(-zrng, 0.0, nz)
# Flip the vector
zgrid_vec = zgrid_vec[::-1]

# Remove variables from memory
del riv_w, dy, dz, ny, nz, zrng, inter, dznew
##########################################
cells = len(xgrid_vec)*len(ygrid_vec)*len(zgrid_vec)
if 15000 > cells and cells >= 10000: print("Building %d Cells:\nExpect a slight delay in figure loading times." % cells)
if 20000 > cells and cells >= 15000: print("Building %d Cells:\nExpect a moderate delay in figure loading times." % cells)
if 200000 > cells and cells >= 20000: print("Building %d Cells:\nExpect a severe delay in figure loading times.\nIt is suggested to increase dx, dy or dz in the config.ini file." % cells)
if cells > 200000: 
    print("The figure loading time to build %d cells is too excessive for viewer performance.\nIncrease dx, dy or dz in the config.ini file." % cells)
    sys.exit(0)

### Define the sliders for the Dashboard ###
# Define the x-slider marks dictionary
xdict = {}
for i, j in enumerate(xgrid_vec):
    xdict[i] = ('%.f' % j)
# Define the y-slider marks dictionary
ydict = {}
for i, j in enumerate(ygrid_vec):
    ydict[i] = ('%.f' % j)
# Define the z-slider marks dictionary
zdict = {}
for i, j in enumerate(zgrid_vec):
    zdict[i] = ('%.2f' % j)
# Remove iterators from memory
del i, j
############################################

### Construct the curvilinear grid and the alpha array ###
# Using the axis vectors and the x-grid's cartesian coordinates, build a curvilinear grid system
xgrid, ygrid, zgrid = curvey_grid(thal_x, thal_y, m_norm, b_norm, xgrid_vec, ygrid_vec, zgrid_vec)
# Get the center index for the ygrid
y_cen = int((xgrid.shape[1] - 1)/2)
# Make a 3D transparency array
alpha = voxelAlpha(xgrid.shape, 1)

# Remove variables from memory
del thal_x, thal_y, m_norm, b_norm
##########################################################

### Construct a trace around the curvey grid for the mapbox ###
# Trim the x, y and z grids down to the outer x and y boundaries and the z boundary to the surface
# This will make a trace around the gridded domain at z = 0 for input into the mapbox
# [0, :, 0] = positions at the upstream most point across the river from left bank to right at the surface
# [:, -1, 0] = positions at the right bank from the upstream to downstream ends of the river at the surface
# flipped [-1, :, 0] = positions at the downstream most point across the river from right bank to left at the surface
# flipped [:, 0, 0] = positions at the left bank from the downstream to upstream ends of the river at the surface
xborder = np.concatenate((xgrid[0, :, 0], xgrid[:, -1, 0], np.flip(xgrid[-1, :, 0]), np.flip(xgrid[:, 0, 0])))
yborder = np.concatenate((ygrid[0, :, 0], ygrid[:, -1, 0], np.flip(ygrid[-1, :, 0]), np.flip(ygrid[:, 0, 0])))

# Get the min and max, x and y points of the domain (outer corners)
xbox = np.asarray([np.min(xgrid), np.max(xgrid), np.max(xgrid), np.min(xgrid)])
ybox = np.asarray([np.min(ygrid), np.min(ygrid), np.max(ygrid), np.max(ygrid)])
# Get the centroid of the domain from the outer corners
xbox_cen, ybox_cen = centroid_2D(xbox, ybox)
# Remove the xbox, ybox from memory
del xbox, ybox
# Build a list of the sequential x/y coordiantes prior to conversion from utm - base station value
border_xy = [[xb, yb] for xb, yb in zip(xborder, yborder)]

# Add back the base x and y values and reconvert the border/centroid points back to meters
xborder = (xborder + base_x)/3.28084
yborder = (yborder + base_y)/3.28084
xbox_cen = (xbox_cen + base_x)/3.28084
ybox_cen = (ybox_cen + base_y)/3.28084

# Now with our units back into meters, transform the points back into lat/lon coordinates
#   This is necessary for the values to be fed into mapbox
lonborder, latborder = p(xborder, yborder, inverse = True)
lonbox_cen, latbox_cen = p(xbox_cen, ybox_cen, inverse = True)

# Build a list of the sequential lat/lon coordinates
border = [[lon, lat] for lon, lat in zip(lonborder, latborder)]

# Remove the utm coordinate variables and the unpackaged lon/lat border variables from memory
del p, zone, base_x, base_y, xborder, yborder, xbox_cen, ybox_cen, lonborder, latborder
################################################

### Build the bathymetry for the 3d plot ###
# Set the x and y grid for interpolating the bathymetry data using the 0th index on kth index (water surface)
bed_x =  xgrid[:, :, 0] 
bed_y = ygrid[:, :, 0] 
# Interpolate the bed to the grid
bed = griddata((x, y), d, (bed_x, bed_y), method='linear')
############################################

### Build the landmarks for the 3d plot ###
if lmc1 == 1:
    lm1 = []
    for lmx, lmy in zip(lm1_x, lm1_y):
        lmx = [lmx]
        lmy = [lmy]
        lmz = griddata((x, y), d, (lmx, lmy), method='linear') + 1.5
        lmz = lmz.tolist()
        dat_bot = go.Scatter3d(
            x = lmx,
            y = lmy,
            z = lmz,
            text = [lm1_text[0]],
            mode = 'markers',
            marker = dict(
                color = 'rgb(255, 178, 44)',
                size = 5,
                symbol = 'x',
                line = dict(
                    color = 'rgb(0, 0, 0)',
                    width = 1
                )
            )
        )
        dat_top = go.Scatter3d(
            x = lmx,
            y = lmy,
            z = [0],
            text = [lm1_text[1]],
            mode = 'markers',
            marker = dict(
                color = 'rgb(255, 178, 44)',
                size = 8,
                symbol = 'circle',
                line = dict(
                    color = 'rgb(0, 0, 0)',
                    width = 1
                )
            )
        )

        lmz = lmz + [0]
        lmx = lmx*2
        lmy = lmy*2
        dat_line = go.Scatter3d(
            x = lmx,
            y = lmy,
            z = lmz,
            mode = 'lines',
            line = dict(
                color = 'rgb(178, 178, 178)',
                width = 5
            ),
            hoverinfo = 'none'
        )
        
        lm1.append(dat_bot)
        lm1.append(dat_top)
        lm1.append(dat_line)


if lmc2 == 1:
    lm2 = []
    for lmx, lmy in zip(lm2_x, lm2_y):
        lmz = griddata((x, y), d, (lmx, lmy), method='linear') + 1.5
        lmz = [lmz.tolist()]
        lmx = [lmx]
        lmy = [lmy]
        dat = go.Scatter3d(
                x = lmx,
                y = lmy,
                z = lmz,
                text = lm2_text[0],
                mode = 'markers',
                marker = dict(
                    color = 'rgb(255, 0, 0)',
                    size = 6,
                    symbol = 'diamond',
                    line = dict(
                        color = 'rgb(0, 0, 0)',
                        width = 1
                    )
                )
            )
        lm2.append(dat)
###########################################

### Setup graph limits and aspect ratios ###
# Get the bounds for x, y and depth
xmin = np.min(xgrid)
xmax = np.max(xgrid)
ymin = np.min(ygrid)
ymax = np.max(ygrid)
dmin = np.min(bed[~np.isnan(bed)])
dmax = np.max(zgrid)
# Prepare the aspect ratios for the 3d figure
asp_x = abs(xmax - xmin)
asp_y = abs(ymax - ymin)
asp_div = max(asp_x, asp_y)
asp_x = asp_x/asp_div
asp_y = asp_y/asp_div
############################################

# Set the slider limits equal to the x, y and z grid shapes
slim_x = xgrid.shape[0]
slim_y = xgrid.shape[1]
slim_z = xgrid.shape[2]
######################################################################

### Load in data for viewer ###
# Css File
css_file = os.path.dirname(os.path.realpath(__file__)) + '\\assets\\4-AUVstylesheet.css'

# Load the CSS file
with open(css_file, 'r') as file:
    # Store the whole file as a string
    css_txt = file.read()

# Use regular expressions to edit the .css slider setup
pat_left = r'^(?P<prefix>(.*\n)*)(?P<keep>#slider-x-parent .rc-slider,[^{]*{([^;|}]*;)*)(?P<edit>\n  left: calc[^;|}]*;\n})(?P<suffix>(.*\n)*})'
pat_width = r'^(?P<prefix>(.*\n)*)(?P<keep>#slider-x-parent .rc-slider-rail,[^{]*{([^;|}]*;)*)(?P<edit>\n  width: calc[^;|}]*;\n})(?P<suffix>(.*\n)*})'
# Adjust the slider rail width and centers the slider within its div based on the number of slider ticks
inj_left = '\n  left: calc(99.9999%% * %.5f);\n}' % (1/((swid - 1)*2))
inj_width = '\n  width: calc(99.9999%% * %.5f);\n}' % ((swid - 2)/(swid - 1))
# Initialize the stringSubReplace object then execute the replacement
ssr = stringSubReplace(css_txt)
ssr.replace([pat_left, pat_width], [inj_left, inj_width])
css_txt = ssr.modified
# Delete the string replacement object and the string variables associated with it
del ssr, pat_left, pat_width, inj_left, inj_width

# Write edits to file
with open(css_file, 'w') as file:
    file.write(css_txt)
###############################

### Build the mapbox figure ###
# Set the mapbox access token from the users mapbox account
mapbox_access_token = 'pk.eyJ1IjoiYmh1ZmZtYW4iLCJhIjoiY2p1eTdtYmk4MHZhdDQ0cHZzeTFjMWl5YyJ9.5G7VBAyq4BN4jdPbb2ieIg'
# Make the layer source for mapbox as a geojson type dictionary
geo_box = {
    # The line trace is a type Feature with empty properties and a geometry type LineString with coordinates set to the sequential lon/lat border list
	"type" : "Feature",
	"properties" : {},
	"geometry" : {
		"type" : "LineString",
		"coordinates" : border
	}
}

# Make the mapbox plotly figure as a geojson type dictionary
topo_fig = {
    # Data assignment for the figure
	"data" : [{
        # Figure type
		"type" : "scattermapbox",
	}],
    # Layout assignment for the figure
	"layout" : {
        # Remove the padding around the mapbox inside the division space
        "margin" : {
            # Left padding
            "l" : 0,
            # Right padding
            "r" : 0,
            # Top padding
            "t" : 0,
            # Bottom padding
            "b" : 0
        },
        # Add the mapbox to the figure
		"mapbox" : {
            # Call the access token
			"accesstoken" : mapbox_access_token,
            # Set the map styling (dark, outdoors, satellite, etc.)
            "style" : "outdoors",
            # Set the center view point coordinates
            "center" : {
                "lon" : lonbox_cen,
                "lat" : latbox_cen
            },
            # Set the bearing, pitch and zoom of the map
            "zoom" : map_zoom,
            "bearing" : map_bearing,
            "pitch" : map_pitch,
			# Import the geojson line trace
            "layers" : [{
				"sourcetype" : "geojson",
				"source" : geo_box,
                # Draw a line with the coordinates
				"type" : "line",
                # Set the line color
				"color" : "rgb(245, 24, 113)"
			}]
		}
	}
}
###############################

### Setup the layouts for the 3d figure ###
# Build the 3d plot layout settings
lay_3d = go.Layout(
    # Set plot font family and size
    font = dict(
        family = 'UniversCond',
        size = 10
    ),
    # Anchor the y axis to the x axis (for equal scaling)
    yaxis = dict(
        scaleanchor = "x", 
        scaleratio = .5
    ),
    # Turn off the plot legend
    showlegend = False,
    # Set a whitespace margin around the figure
    margin = dict(
        l = marg3d,
        r = marg3d,
        t = marg3d,
        b = marg3d
    ),
    # Set the x, y and z axis titles and ranges
    scene = dict(
        xaxis = dict(
            title = 'Easting (ft)',
            range = [xmin, xmax]
        ),
        yaxis = dict(
            title = 'Northing (ft)',
            range = [ymin, ymax]
        ),
        zaxis = dict(
            title = 'Depth (ft)',
            range = [dmin, dmax]
        ),
        # Set the aspect ratio for x, y and z (z is independent while x and y are linked so they show as equal in the figure)
        aspectmode = "manual",
        aspectratio = dict(
            x = asp_x, 
            y = asp_y, 
            z = (asp_z*(abs(np.max(zgrid) - np.min(zgrid))/asp_div))
        )
    )
)
###########################################

### Build the 2D figure layouts ###
## Make bounds, tickvals and tick text for thalwag axii (x)
# Set a lower and upper bound
xlb = np.min(xgrid_vec)
xub = np.max(xgrid_vec)
# Get the "dx" value from the x grid
xdelta = int(abs(xgrid_vec[1] - xgrid_vec[0]))
x_rng = abs(xub - xlb)
# Check if there are more than 20 x ticks and if so ...
if (x_rng/xdelta) > 20:
    # Try different delta's to reduce the number of tick marks
    if (x_rng/50) <= 20:
        xdelta = 50
    if (x_rng/100) <= 20:
        xdelta = 100
    elif (x_rng/200) <= 20:
        xdelta = 200
    elif (x_rng/500) <= 20:
        xdelta = 500
    xtv = [i for i in range(int(xlb), (int(xub) + int(xdelta) + 1), int(xdelta))]    
else:
    xtv = [i for i in range(int(xlb), (int(xub) + 1), int(xdelta))]
# Generate the x tick text
xtt = [str(i) for i in xtv]

## Make bounds, tickvals and tick text for cross-track axii (y)
# Set a lower and upper bound
ylb = np.min(ygrid_vec)
yub = np.max(ygrid_vec)
# Get the "dy" value from the y grid
ydelta = int(abs(ygrid_vec[1] - ygrid_vec[0]))
y_rng = abs(yub - ylb)
# Check if there are more than 20 x ticks and if so ...
if (y_rng/ydelta) > 20:
    # Try different delta's to reduce the number of tick marks
    if (y_rng/50) <= 20:
        ydelta = 50
    if (y_rng/100) <= 20:
        ydelta = 100
    elif (y_rng/200) <= 20:
        ydelta = 200
    elif (y_rng/500) <= 20:
        ydelta = 500
    ytv = [i for i in range((int(ylb) - int(ydelta)), (int(yub) + int(ydelta) + 1), int(ydelta))]
else:
    ytv = [i for i in range(int(ylb), (int(yub) + 1), int(ydelta))]
# Generate the y tick text
ytt = [str(i) for i in ytv]

## Make bounds, tickvals and tick text for depth axii (z)
# Set a lower and upper bound (make the ticks space by 5 feet)
dlb = round(min(d)/5)*5
dub = 0
# Generate the z tick values and text
dtv = [i for i in range(int(dlb), (int(dub) + 1), 5)]
dtt = [str(i) for i in dtv]
# Set the hover info data appear to 2 decimal places
hform = '.2f'

# Note: If an axis needs to be flipped switch the order of the upper bound and lower bound when feeding the lay_2d function
# X-figure layout
lay_2dx = lay_2d('Cross-Sectional View', 'Distance from Centerline (ft)', 'Depth (ft)', marg, [ylb, yub], ytv, ytt, [dlb, dub], dtv, dtt, hform)
# Y-figure layout
lay_2dy = lay_2d('Longitudinal Profile', 'Distance Downstream (ft)', 'Depth (ft)', marg, [xlb, xub], xtv, xtt, [dlb, dub], dtv, dtt, hform)
# Z-figure layout
lay_2dz = lay_2d('Overhead View', 'Distance Downstream (ft)', 'Distance from Centerline (ft)', marg, [xlb, xub], xtv, xtt, [yub, ylb], ytv, ytt, hform)
###################################

### Step 4: Map data to the curvey grid and make 3d surface plots for data types ###
# Initialize the master storage lists
master_var = []
master_vargrid = []
master_label = []
master_unit = []
master_unit_short = []
master_reso = []
master_colorbar = []
# Create a case_load object
beta = case_load()

# Loop over the data types provided in PltVar
for pv in PltVar:
    try:
        # Call the .get_dfvar function property of the object to get the var (variable data), map (the colormap) and the bounds (value bounds for the colormap)
        var, label, unit, unit_short, map, comp_map, bounds, reso = beta.get_dfvar(pv, dat_df)
        # Filter the data to the waypoints of interest
        var = var #[idx_s:idx_e]

        # Interpolate data to the grid
        vargrid = griddata((x, y, z), var, (xgrid, ygrid, zgrid), method = 'linear')
        
        ### Limit the data values to above the bed surface
        # Grab the grid shape
        ii, jj, kk = vargrid.shape 
        # Loop over the grid
        for i in range(0, ii):
            for j in range(0, jj):
                # If the interpolated bed surface is nan
                if np.isnan(bed[i, j]):
                    # Set all the parameter data at this x, y index as nan
                    vargrid[i, j, :] = np.nan
                # If the interpolated bed surface has a value
                else:
                    # Loop over the depth index
                    for k in range(0, kk):
                        # Find where its lower than the bed surface
                        if zgrid[i, j, k] < bed[i, j]:
                            # Go up one index in the depth and set all parameter data below as nan
                            vargrid[i, j, (k - 1):] = np.nan
                            # Exit the loop over the depth index (k)
                            break
        
        # Grab the shape of the auv vector data
        mm = x.shape[0]
        # Loop over the grid
        for i in range(0, ii):
            for j in range(0, jj):
                # Find the smallest euclidean distance, in R2, between the current grid cell
                #   and the AUV path
                xdiff = x - xgrid[i, j, 0]
                xdiff2 = ms(xdiff)
                ydiff = y - ygrid[i, j, 0]
                ydiff2 = ms(ydiff)
                xydist2 = xdiff2 + ydiff2
                far = np.min(msqrt(xydist2))

                if far > 200:
                    bed[i, j] = np.nan

                for k in range(0, kk):
                    # Find the smallest euclidean distance, in R3, between the current grid cell
                    #   and the AUV path
                    xdiff = x - xgrid[i, j, k]
                    xdiff2 = ms(xdiff)
                    ydiff = y - ygrid[i, j, k]
                    ydiff2 = ms(ydiff)
                    zdiff = z - zgrid[i, j, k]
                    zdiff2 = ms(zdiff)
                    xyzdist2 = xdiff2 + ydiff2 + zdiff2
                    far = np.min(msqrt(xyzdist2))

                    # If the path was further than 110 ft from this cell, set the data to nan
                    if far > 110:
                        vargrid[i, j, k] = np.nan

        # Build colorscale and colorbar objects for plotly figures
        cb_scale_mpl, cb_tickval, cb_ticktext, cb_scale_pltly = generateColorbar(map, bounds)

        # Store the vargrid data, labels, colorbar controls and the artists/graphic objects
        master_var.append(var)
        master_vargrid.append(vargrid)
        master_label.append(label)
        master_unit.append(unit)
        master_unit_short.append(unit_short)
        master_reso.append([reso])
        master_colorbar.append([cb_scale_mpl, cb_tickval, cb_ticktext, cb_scale_pltly, comp_map])

        # Clear memory of variables used in the loop (no longer necessary)
        del pv, label, map, bounds, cb_scale_mpl, cb_tickval, cb_ticktext, cb_scale_pltly #, idx_s, idx_e

    # Parameter is not in the data set
    except:
        pass

del dat_df, beta

# Make the bed surface graphic object
bed_surf = go.Surface(
               x = bed_x,
               y = bed_y,
               z = bed,
               showscale = False
           )

### Build Dashboard ###
# Initialize the dash dashboard object
app = Dash(__name__)

# Build the layout
#   When building the layout we used bootstrap which can take classNames such
#     as 'row' and 'col' or 'col-[1-12]'
#   Each bootstrap row contains 12 columns which can be nested for ex:
#     We make a row then use two div's inside with one col-3 and another col-9
#     Then in col-3 we make another row with three divs inside with each col-4
#   Also in each div we have either a className or id to point to that division either
#     with bootstrap, our .css file or app callbacks
#   Each Div should have a child (children) but not every child will hold a Div
#   HTML Elements we use are Div, H1, H3, P, Span, A and Img
#   Dash Core Components (dcc) we use are Graph, Dropdown, Checklist and Slider
#   Within a dcc if all that is listed is the id that is okay as the callbacks below the
#     app.layout setup will run and populate those dcc objects
app.layout = html.Div([
    # Landing Header
    html.Div(
        # Use the class name 'head' for the .css
		className = 'head',
		# Main title text
        children = [
			html.H1('Waterbody Rapid Assessment Tool (Water RAT)', className = 'head-text')
		]
	),
	
	# Sub landing paragraph
	html.Div(
        # Use the class name 'section' for the .css
		className = 'section',
        # Brief discription text
		children = [
			html.P(dedent('''This tool visualizes 3D data captured by an autonomous underwater vehicle (AUV) deployed by the United States Geological Survey.  
					The top left map shows the study area. The top right figure represents water quality data in three dimensions. Select the water uality parameter of interest from the dropdown box.
                    The three bottom figures represent water quality data in two dimensionsional slices. Use the sliders to translate along each dimension and view different slices of the three dimensional space.'''),
					className = 'section-text')
		]
	),

    # Content division
    html.Div(
        className = 'row',
        # Use the id 'content' for the .css
        id = 'content',
        children = [
            html.Div(
                # Make the division XX columns wide (out of 12)
                className = 'col-10',
                id = 'content-upper-graphs',
                # This content division holds all the figures and sliders
                children = [
                    # Upper figures (left: map, above right: parameter dropdown and plot controls, below right: 3d figure)
                    html.Div(
                        # Use the id for the .css
                        id = 'upper-graphs',
                        # Row containing the upper figures
                        className = 'row',
                        children = [
                            html.Div(
                                # Column split for the map figure
                                className = 'col-3',
                                # Use id for the .css
                                id = 'topo-box',
                                children = [
                                    # Title over map
                                    html.Span('Study Area'),
                                    # Map figure
                                    dcc.Graph(
                                        # Use id for the .css
                                        id = 'topo-graph',
                                        figure = topo_fig,
                                        config = dict(
                                            scrollZoom = True
                                        )
                                    )
                                ]
                            ),
                            html.Div(
                                # Column split for the 3d figure, param dropdown and checklist plot items
                                className = 'col-9',
                                children = [
                                    html.Div(
                                        # Row holding the dropdown and checklist items
                                        className = 'row',
                                        children = [
                                            html.Div(
                                                className = 'col-4',
                                                children = [
                                                    # Parameter dropdown box
                                                    dcc.Dropdown(
                                                        # Use id for .css and for callbacks
                                                        id = 'main-param',
                                                        # Make the options dictionary from master_label (indexed with master_ax and master_vargrid)
                                                        options = [{'label' : param, 'value' : i} for i, param in enumerate(master_label)],
                                                        # Set the first param to the zero index
                                                        value = 0
                                                    )
                                                ]
                                            ),
                                            # Checklist
                                            dcc.Checklist(
                                                # Use id for .css and callbacks
                                                id = 'checkboxes',
                                                className = 'col-5',
                                                # Make the options dictionary for model wireframe, auv track and bed depth
                                                # Use prime number values for the options (for summing to unique cases)
                                                options = [
                                                    {'label': 'Modeled Region', 'value': '3'},
                                                    {'label': 'AUV Track', 'value': '5'},
                                                    {'label': 'Bathymetry', 'value': '7'}
                                                ],
                                                # Set to empty initially (none of the options displayed)
                                                values = []
                                            ),
                                            html.Div(
                                                className = 'col-3',
                                                children = [
                                                    html.Div(
                                                        className = 'row',
                                                        id = 'aspect_z',
                                                        children = [
                                                            html.Div(
                                                                className = 'col-9',
                                                                children = [
                                                                    html.P(
                                                                        'Vertical Aspect Ratio :'
                                                                    )
                                                                ]
                                                            ),
                                                            html.Div(
                                                                className = 'col-3',
                                                                children = [
                                                                    dcc.Input(
                                                                        id = 'aspect_z_input', 
                                                                        type = 'number',
                                                                        inputmode = 'numeric',
                                                                        debounce = True,
                                                                        value = asp_z
                                                                    ) 
                                                                ]
                                                            )
                                                        ]
                                                    )
                                                ]
					    )
                                        ]
                                    ),
                                    html.Div(
                                        # Row containing the 3d figure
                                        className = 'row',
                                        children = [
                                            html.Div(
                                                className = 'col-12',
                                                children = [
                                                    dcc.Loading(
                                                        id = 'loading-main-graph',
                                                        type = 'graph',
                                                        children = [
                                                            # 3d Figure
                                                            dcc.Graph(
                                                                # Use id for .css
                                                                id = 'main-graph',
                                                                config = dict(
                                                                    scrollZoom = True
                                                                )
                                                            )
                                                        ]
                                                    )
                                                ]
                                            )
                                        ]
                                    )  
                                ]
                            )
                        ]
                    ),
                    # Slider Row division
                    html.Div(
                        # Row containing the sliders
                        className = 'row',
                        # Set an id for .css
                        id = 'slider-row',
                        children = [
                            html.Div(
                                # Column split for the x-figure slider
                                className = 'col-4',
                                # Use id for .css
                                id = 'slider-x-parent',
                                children = [
                                    # Header above the slider
                                    html.H3(
                                        'Distance Downstream (ft)',
                                        # Use className for .css
                                        className = 'slider-label'
                                    ),
                                    # Dynamic Slider
                                    dcc.Slider(
                                        # Use id for .css and callback
                                        id = 'slider-x',
                                        # Set the initial min and max indicies to display (linked with the marks dictionary)
                                        min = 0, 
                                        max = swid,
                                        # Set the initial value to 0 (beginning of the reach)
                                        value = 0,
                                        # Step set to none, its unnecessary
                                        step = None, 
                                        # Turn off the trailing rail highlight
                                        included = False,
                                        # Set the marks dictionary
                                        marks = dict(list(xdict.items())[0:swid])
                                    )
                                ]
                            ),
                            html.Div(
                                # Column split for the y-figure slider
                                className = 'col-4',
                                # Use id for .css
                                id = 'slider-y-parent',
                                children = [
                                    # Header above the slider
                                    html.H3(
                                        'Distance from Centerline (ft)',
                                        # Use className for .css
                                        className = 'slider-label'
                                    ),
                                    # Static Slider
                                    dcc.Slider(
                                        # Use id for .css and callback
                                        id = 'slider-y',
                                        # Set the initial min and max indicies to display (linked with the marks dictionary)
                                        min = int(y_cen - (swid/2)), 
                                        max = int(y_cen + (swid/2) + 1), 
                                        # Set the initial value to y_cen (centerline of the stream area)
                                        value = y_cen, 
                                        # Step set to none, its unnecessary
                                        step = None,
                                        # Turn off the trailing rail highlight
                                        included = False,
                                        # Set the marks dictionary
                                        marks = dict(list(ydict.items())[int(y_cen - (swid/2)):int(y_cen + (swid/2) + 1)])
                                    )
                                ]
                            ),
                            html.Div(
                                # Column split for the z-figure slider
                                className = 'col-4',
                                # Use id for .css
                                id = 'slider-z-parent',
                                children = [
                                    # Header above the slider
                                    html.H3(
                                        'Depth (ft)',
                                        # Use className for .css
                                        className = 'slider-label'                            
                                    ),
                                    # Dynamic Slider
                                    dcc.Slider(
                                        # Use id for .css and callback
                                        id = 'slider-z',
                                        # Set the initial min and max indicies to display (linked with the marks dictionary)
                                        min = 0, 
                                        max = swid, 
                                        # Set the initial value to 0 (water surface)
                                        value = 1, 
                                        # Step set to none, its unnecessary
                                        step = None,
                                        # Turn off the trailing rail highlight
                                        included = False,
                                        # Set the marks dictionary
                                        marks = dict(list(zdict.items())[0:swid])
                                    )
                                ]
                            )
                        ]
                    ),             
                    # 2D Figures row
                    html.Div(
                        # Use id for .css
                        id = 'lower-graphs',
                        # Row containing split columns for each of the three figures
                        className = 'row',
                        children = [
                            html.Div(
                                # We are spliting the row into 3 columns so 12/3 = 4, ergo col-4
                                className = 'col-4',
                                children = [
                                    # X figure 
                                    dcc.Graph(
                                        # Use id for callbacks
                                        id = '2dx'
                                    )
                                ]
                            ),
                            html.Div(
                                className = 'col-4',
                                children = [
                                    # Y figure
                                    dcc.Graph(
                                        # Use id for callbacks
                                        id = '2dy'
                                    )
                                ]
                            ),
                            html.Div(
                                className = 'col-4',
                                children = [
                                    # Z figure
                                    dcc.Graph(
                                        # Use id for callbacks
                                        id = '2dz'
                                    )
                                ]
                            )
                        ]
                    ),                    
                    # About row
                    html.Div(
                        # Row containing employee title and name w/ link and SAWSC link and USGS logo
                        className = 'row',
                        # Use id for .css
                        id = 'about-row',
                        children = [
                            html.Div(
                                # Column to hold employee titles and names w/ links
                                className = 'col-6',
                                children = [
                                    html.Div(
                                        # Row to contain split columns for the three employees
                                        className = 'row',
                                        # Use id for .css
                                        id = 'developers',
                                        children = [
                                            html.Div(
                                                className = 'col-4',
                                                children = [
                                                    # HTML p tag for text next to a link
                                                    html.P(
                                                        'Project Chief:  '
                                                    ),
                                                    # HTML a tag for text with a hyperlink
                                                    html.A(
                                                        'Jimmy M. Clark',
                                                        href = 'https://www.usgs.gov/staff-profiles/jimmy-m-clark',
                                                        target = '_blank'
                                                    )
                                                ]
                                            ),
                                            html.Div(
                                                className = 'col-4',
                                                children = [
                                                    # HTML p tag for text next to a link
                                                    html.P(
                                                        'Developer:  '
                                                    ),
                                                    # HTML a tag for text with a hyperlink
                                                    html.A(
                                                        'Brad J. Huffman',
                                                        href = 'https://www.usgs.gov/staff-profiles/brad-j-huffman',
                                                        target = '_blank'
                                                    )
                                                ]
                                            ),
                                            html.Div(
                                                className = 'col-4',
                                                children = [
                                                    # HTML p tag for text next to a link
                                                    html.P(
                                                        'Developer:  '
                                                    ),
                                                    # HTML a tag for text with a hyperlink
                                                    html.A(
                                                        'Andrea S. Medenblik',
                                                        href = 'https://www.usgs.gov/staff-profiles/andrea-s-medenblik',
                                                        target = '_blank'
                                                    )
                                                ]
                                            )
                                        ]
                                    )
                                ]
                            ),
                            html.Div(
                                # Column to hold the SAWSC link and USGS logo
                                className = 'col-6',
                                children = [
                                    html.Div(
                                        # Row containing the USGS logo
                                        className = 'row',
                                        # Use id for .css
                                        id = 'sawsc-img',
                                        children = [
                                            # HTML img tag with link to a usgs logo
                                            html.Img(
                                                src = "https://www.usgs.gov/sites/all/themes/usgs_palladium/logo.png"
                                            )
                                        ]
                                    ),
                                    html.Div(
                                        # Row containing the SAWSC link
                                        className = 'row',
                                        # Use id for .css
                                        id = 'sawsc-link',
                                        children = [
                                            # HTML a tag for text with hyperlink
                                            html.A(
                                                'USGS South Atlantic Water Science Center',
                                                href = 'https://www.usgs.gov/centers/sa-water',
                                                target = '_blank'
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            ),
            html.Div(
                # Column to contain the master colorbar
                className = 'col-2',
                # Use id for .css
                id = "content-colorbar",
                children = [
                    html.Div(
                        # Row to contain the commit button
                        className = 'row',
                        children = [
                            html.Div(
                                className = 'col-6',
                                children = [
                                    dcc.Textarea(
                                        id = 'valid-prompt',
                                        value = '',
                                        disabled = True,
                                        wrap = True,
                                        hidden = True,
                                        rows = 4
                                    )
                                ]
                            ),
                            html.Div(
                                className = 'col-6',
                                children = [
                                    html.Button('Update Bounds', id = 'commit-button')
                                ]
                            )
                        ]
                    ),
                    html.Div(
                        # Row to contain the master colorbar upper bound adjustment
                        className = 'row',
                        id = 'up-bound',
                        children = [
                            html.Div(
                                className = 'col-6',
                                children = [
                                    html.P(
                                        'Upper Bound :'
                                    )
                                ]
                            ),
                            html.Div(
                                className = 'col-6',
                                children = [
                                    # Colorbar upper bound input
                                    dcc.Input(
                                        # Hard settings
                                        id = 'cbar-ubound', 
                                        type = 'number',
                                        inputmode = 'numeric',
                                        debounce = True
                                    ) 
                                ]
                            )
                        ]
                    ),
                    html.Div(
                        # Row to contain the master colorbar
                        className = 'row',
                        id = 'cbar-img',
                        children = [
                            # HTML image tag for the URI converted matplotlib colorbar
                            html.Img(
                                # Use id for callbacks
                                id = "cbar",
                                src = ""
                            )
                        ]
                    ),
                    html.Div(
                        # Row to contain the master colorbar lower bound adjustment
                        className = 'row',
                        id = 'lo-bound',
                        children = [
                            html.Div(
                                className = 'col-6',
                                children = [
                                    html.P(
                                        'Lower Bound :'
                                    )
                                ]
                            ),
                            html.Div(
                                className = 'col-6',
                                children = [
                                    # Colorbar lower bound input
                                    dcc.Input(
                                        # Hard settings
                                        id = 'cbar-lbound', 
                                        type = 'number',
                                        inputmode = 'numeric',
                                        debounce = True
                                    )
                                ]
                            )
                        ]
                    )
                ]
            )
        ]
    )
])

#       The following section deals with callbacks within the dashboard/webpage
#   when @app.callback is listed.  It works as follows: you assign the 
#   @app.callback function to inputs and outputs based on the id given to the 
#   objects of interest. As the program runs, it will monitor for changes in
#   the specified inputs.  
#       The monitored items within those objects can be referenced
#   within each dashIn function. For example, a slider has attributes
#   like min, max, value, step, etc. So dashIn(slider_id, value) would get the value from
#   a slider with that specific id. Once a change is noticed the callback will direct
#   to a def function directly below syntactically with the current input values, where 
#   a programmed task can take place. State variables can be used to capture
#   current properties of objects, without having their changes trigger the callback, denoted 
#   by dashState().  
#       Once the task is completed, you return the type of object refered to in the dashOut 
#   function inside the @app.callback. This will update the dashboard/webpage with the new 
#   object. To output multiple properties, use a list around the multiple dashOut functions.

# Callback to set the colorbar upper/lower bound initial values, step increments and 
#   the lower bound minimum input value
@app.callback(
    [dashOut('cbar-ubound', 'value'), dashOut('cbar-lbound', 'value'), dashOut('cbar-ubound', 'step'), dashOut('cbar-lbound', 'step')],
    [dashIn('main-param', 'value')]
)
def paramChange_inputBoundSetup(param_idx):
    ini_ubound = master_colorbar[param_idx][1][-1]
    ini_lbound = master_colorbar[param_idx][1][0]

    if len(master_reso[param_idx][0]) == 1:
        step = master_reso[param_idx][0][0]
    else:
        if (master_reso[param_idx][0][1]*ini_ubound) > master_reso[param_idx][0][0]:
            step = (master_reso[param_idx][0][1]*ini_ubound)
        else:
            step = master_reso[param_idx][0][0]

    return ini_ubound, ini_lbound, step, step

# Callback to reset the number of clicks for the "Update Bounds" button
@app.callback(
    dashOut('commit-button', 'n_clicks'),
    [dashIn('main-param', 'value')]
)
def paramChange_resetClicks(_):
    return 0

# Callback to provide logic control on the colorbar upper and lower bound input values
#   along with prompting the user if and what the issue might be
@app.callback(
    [dashOut('valid-prompt', 'value'), dashOut('valid-prompt', 'hidden')],
    [dashIn('commit-button', 'n_clicks')],
    state = [dashState('main-param', 'value'), dashState('cbar-ubound', 'value'), dashState('cbar-lbound', 'value'), dashState('cbar-ubound', 'step')]
)
def validate_inputs(click, param_idx, cb_upper, cb_lower, step):
    if (click != None) and (cb_upper != None) and (cb_lower != None) and (step != None):
        intvl = len(master_colorbar[param_idx][0])
        if (cb_lower <= (cb_upper - ((intvl - 2)*step))) and (cb_upper >= (cb_lower + ((intvl - 2)*step))):
            message = ''
        elif (cb_lower > cb_upper):
            message = 'The upper bound must not be less than the lower bound.'
        elif (cb_lower == cb_upper):
            message = 'The upper and lower bound must not be equal to each other.' 
        else:
            message = 'The upper and lower bound must not be within ' + '%.2f' % ((intvl - 2)*step) + ' units from each other.'
    else:
        raise dashNoUpdate

    if message == '':
        show = True
    else:
        show = False

    return message, show

# Callback to update the colorbar 
@app.callback(
    dashOut('cbar', 'src'),
    [dashIn('main-param', 'value'), dashIn('valid-prompt', 'value')],
    state = [dashState('cbar-ubound', 'value'), dashState('cbar-lbound', 'value'), dashState('commit-button', 'n_clicks')]
)
def update_colorbar(param_idx, message, cb_upper, cb_lower, click):
    if message == '':
        plt.style.use('dark_background')

        # Make a figure and axes with dimensions as desired.
        fig, ax = plt.subplots(figsize = (2, 7.5))

        # Internal colormap
        cmap = ListedColormap(master_colorbar[param_idx][0][1:-1])

        # External color maps
        cmap.set_over(tuple(master_colorbar[param_idx][0][-1]))
        cmap.set_under(tuple(master_colorbar[param_idx][0][0]))
        
        # Set the value boundaries and make the labels
        if ((cb_upper != None) and (cb_lower != None)) and ((cb_upper != master_colorbar[param_idx][1][-1]) or (cb_lower != master_colorbar[param_idx][1][0])) and (click != 0):
            bounds = np.linspace(cb_lower, cb_upper, num = int(len(master_colorbar[param_idx][1]))).tolist()
        else:
            bounds = master_colorbar[param_idx][1]
        labels = tickval2text(bounds)

        # Set the normal boundary using the colormap and bounds
        norm = BoundaryNorm(bounds, cmap.N)

        # Define the colorbar
        cb = ColorbarBase(ax, cmap = cmap,
                                        norm = norm,
                                        boundaries = [0] + bounds + [999999],
                                        extend = 'both',
                                        # Make the length of each extension
                                        # the same as the length of the
                                        # interior colors:
                                        extendfrac = 'auto',
                                        ticks = bounds,
                                        spacing = 'uniform',
                                        orientation = 'vertical')

        # Add the label and set the background to black
        cb.set_label(master_unit[param_idx])
        cb.set_ticklabels(labels)

        # Adjust the layout
        plt.tight_layout()

        # Get the final figure
        fig = plt.gcf()
        # Convert to uri
        figURI = mplToURI(fig)

        return figURI

# Updates Main 3D figure when parameter dropbox or checkboxes change
@app.callback(
    dashOut('main-graph', 'figure'),
    [dashIn('main-param', 'value'), dashIn('checkboxes', 'values'), dashIn('cbar', 'src'), dashIn('aspect_z_input', 'value')],
    state = [dashState('cbar-ubound', 'value'), dashState('cbar-lbound', 'value')]
)
def update_3d_traces(param_idx, checkbox_values, cbimg, aspect_z, cb_upper, cb_lower):
    if cbimg != None:
        # Turn all the current checkbox values into integers
        checkbox_values = [int(s) for s in checkbox_values]
        # Sum up the checkbox values (since values are prime a unique value will exist)
        sum_check = np.sum(checkbox_values)

        # Initialize the data list for the plotly figure
        dat1 = []

        # Check if the unique sum is any of the below which can only exist if the modeled region frame is selected
        if (sum_check == 3) or (sum_check == 8) or (sum_check == 10) or (sum_check == 15):
            # Set the value boundaries and make the labels
            if ((cb_upper != None) and (cb_lower != None)) and ((cb_upper != master_colorbar[param_idx][1][-1]) or (cb_lower != master_colorbar[param_idx][1][0])):
                bounds = np.linspace(cb_lower, cb_upper, num = int(len(master_colorbar[param_idx][1]))).tolist()
            else:
                bounds = master_colorbar[param_idx][1]

            # Add the zero lower limit back in for use with voxelMat
            bounds = [0] + bounds

            # Initialize the axis artist
            ax = []
            # Make poly surfaces
            dat1 = voxelMat(ax, xgrid, ygrid, zgrid, master_vargrid[param_idx], alpha, np.asarray(master_colorbar[param_idx][0]), bounds)
            # Remove ax from memory
            del ax

            # Extract the x and y positions from the model frame border
            xb = [b[0] for b in border_xy]
            yb = [b[1] for b in border_xy]
            # Set the z positions as zeros (must be equal vector length as xb and yb)
            zb = list(np.zeros(len(xb)))
            # Make the scatter3d trace
            dat2 = go.Scatter3d(
                # Assign the border points 
                x = xb,
                y = yb,
                z = zb,
                # Set the scatter as a line
                mode = 'lines',
                # Set line properties
                line = dict(
                    color = 'rgb(255, 0, 0)',
                    width = 3
                ),
                # Give the line a hover name
                name = 'Modeled Region',
                hoverinfo = 'name'          
            )
            # Append this trace to the copied data
            dat1.append(dat2)

        # Check if the unique sum is any of the below which can only exist if the AUV track is selected
        if (sum_check == 8) or (sum_check == 15):
            rgb = master_colorbar[param_idx][4]
            c = 'rgb(%3.f, %3.f, %3.f)' % (rgb[0], rgb[1], rgb[2])

            # Make the scatter3d trace
            dat3 = go.Scatter3d(
                # Assign the border points 
                x = x,
                y = y,
                z = z,
                # Set the scatter as a line
                mode = 'lines',
                # Set line properties
                line = dict(
                    color = c,
                    width = 7
                ),
                # Give the line a hover name
                name = 'AUV Track',
                hoverinfo = 'name'          
            )

            # Append this trace to the copied data
            dat1.append(dat3)

        elif (sum_check == 5) or (sum_check == 12):
            # Set the value boundaries and make the labels
            if ((cb_upper != None) and (cb_lower != None)) and ((cb_upper != master_colorbar[param_idx][1][-1]) or (cb_lower != master_colorbar[param_idx][1][0])):
                bounds = np.linspace(cb_lower, cb_upper, num = int(len(master_colorbar[param_idx][1]))).tolist()
            else:
                bounds = master_colorbar[param_idx][1]

            # Add the zero lower limit back in for use with voxelMat
            bounds = [0] + bounds
            
            # Bin the data by indices corresponding to the boundaries given
            bin_idx = indexByBin(master_var[param_idx], bounds)
            # Convert the colormap from base 0-1 to 0-255
            cscale = ['rgb(%3.f, %3.f, %3.f)' % ((rgb_map[0]*255), (rgb_map[1]*255), (rgb_map[2]*255)) for rgb_map in master_colorbar[param_idx][0]]
            # Make x and y into lists
            ax = x.tolist()
            ay = y.tolist()

            # Loop over the colorscales and the linked binned indices
            for cs, idx in zip(cscale, bin_idx):
                # Unpack the indice list
                ii = idx[0]
                # Verify the list isnt empty
                if len(ii) != 0:
                    # Make a boolean vector the same length as the data thats True everywhere
                    mask = np.ones(len(master_var[param_idx]), dtype = bool)

                    # Add the neighboring indices for the first and last elements of the indice list
                    ii = [ii[0] - 1] + ii + [ii[-1] + 1]
                    # Add the neighboring indices for breaks/jumps of more than one index to the indice list
                    i_add = []
                    start = 0
                    for j in range(1, len(ii)):
                        if ((ii[j] - ii[j - 1]) != 1):
                            i_add  = i_add + ii[start:j] + [ii[j - 1] + 1, ii[j] - 1]
                            start = j
                    i_add = i_add + ii[start:]

                    # Convert the modified indice list to a numpy array
                    ii = np.asarray(i_add)
                    # Remove the possiblity of having indices outside the index range of the data
                    ii = np.delete(ii, np.where(ii < 0)[0])
                    ii = np.delete(ii, np.where(ii >= len(mask))[0])

                    # Set the elements in mask to False at the indices where we want to color our data
                    mask[ii] = False
                    # Copy z to az
                    az = np.copy(z)
                    # Using the elements where mask is True (where we dont want to color our data), set
                    #   the values of az to None so plotly will ignore them in plotting
                    az[mask] = None
                    # Make az a list
                    az = az.tolist()
                    
                    # Make the 3d line plot for the data colored by this specific color
                    dat3 = go.Scatter3d(
                        x = ax,
                        y = ay,
                        z = az,
                        mode = 'lines',
                        line = dict(
                            color = cs,
                            width = 7
                        )
                    )

                    # Add this line plot to the graphic object dictionary
                    dat1.append(dat3)

        # Check if the unique sum is any of the below which can only exist if the bathymetry is selected
        if (sum_check == 7) or (sum_check == 10) or (sum_check == 12) or (sum_check == 15):
            # Add the bed surface
            dat1.append(bed_surf)

        # Add landmarks
        if lmc1 == 1:
            for landmark in lm1:
                dat1.append(landmark)
        if lmc2 == 1:
            for landmark in lm2:
                dat1.append(landmark)
		
        # Get the current vertical aspect ratio
        lay_3d.scene.aspectratio.z = (aspect_z*(abs(np.max(zgrid) - np.min(zgrid))/asp_div))

        # Return the 3d figure w/ the current parameter and checkbox options
        return go.Figure(data = dat1, layout = lay_3d)
    
    else:
        raise dashNoUpdate

## Below are the callbacks for updating the lower figures when using the sliders ##

# Updates Cross-Section Figure when the x slider changes
@app.callback(
    dashOut('2dx', 'figure'),
    [dashIn('slider-x', 'value'), dashIn('main-param', 'value'), dashIn('cbar', 'src')],
    state = [dashState('cbar-ubound', 'value'), dashState('cbar-lbound', 'value')]
)
def update_xfig_xslider(x_idx, param_idx, cbimg, cb_upper, cb_lower):
    if cbimg != None:
        # Set the color value boundaries and step size
        if ((cb_upper != None) and (cb_lower != None)) and ((cb_upper != master_colorbar[param_idx][1][-1]) or (cb_lower != master_colorbar[param_idx][1][0])):
            bounds = np.linspace(cb_lower, cb_upper, num = int(len(master_colorbar[param_idx][1]))).tolist()
        else:
            bounds = master_colorbar[param_idx][1]
        step_size = bounds[1] - bounds[0]

        # Make the contour trace
        dat1 = go.Contour(
            # Get the data matrix for the contour using the current parameter and longitudinal index
            #   from the x slider
            # Transpose the matrix
            z = np.transpose(master_vargrid[param_idx][x_idx, :, :]),
            # Define the horizontal axis starting point and spacing (cross-section)
            dx = abs(ygrid_vec[1] - ygrid_vec[0]),
            x0 = ygrid_vec[0],
            # Define the vertical axis starting point and spacing (depth)
            dy = (-1*abs(zgrid_vec[1] - zgrid_vec[0])),
            y0 = zgrid_vec[0],
            # Set the hover data name
            name = '<br><br>'+ master_label[param_idx],
            hoverinfo = 'x+y+z+name',
            # Turn off autocontouring and define the levels below
            autocontour = False,
            contours = dict(
                # Set to type levels (constraints can't handle multiple arbitrary levels)
                #   Limitation of levels is it must be equal spacing
                type = 'levels',
                # Set the colorbar properties from the master_colorbar given the current parameter
                # Set the lower value
                start = bounds[0],
                # Set the upper value
                end = bounds[-1],
                # Set the spacing
                size = step_size,
                # Remove the dividing lines on the colorbar
                showlines = False
            ),
            # Turn off the autocolorscale and define the colors to use below
            autocolorscale = False,
            # Set the colors corresponding to the previously defined levels
            colorscale = master_colorbar[param_idx][3],
            # Show the colorbar
            showscale = False
        )

        # Make the scatter trace for the bed
        dat2 = go.Scatter(
            # Assign the bed depth across the river
            x = ygrid_vec,
            y = bed[x_idx, :],
            # Set the hover name as Depth with the actual value
            name = 'Bathymetry',
            hoverinfo = 'y+name',
            # Use a line and marker for the bed depth
            mode = 'lines+markers',
            # Marker properties
            marker = dict(
                size = 5,
                color = 'rgba(0, 0, 0, .8)'
            ),
            # Line properties
            line = dict(
                width = 2,
                color = 'rgba(0, 0, 0, .8)'
            )
        )

        # Return the figure with the contour trace and bed depth trace together
        return go.Figure(data = [dat1, dat2], layout = lay_2dx)

    else:
        raise dashNoUpdate

# Updates Longitudinal Figure when the y slider changes
@app.callback(
    dashOut('2dy', 'figure'),
    [dashIn('slider-y', 'value'), dashIn('main-param', 'value'), dashIn('cbar', 'src')],
    state = [dashState('cbar-ubound', 'value'), dashState('cbar-lbound', 'value')]
)
def update_yfig(y_idx, param_idx, cbimg, cb_upper, cb_lower):
    if cbimg != None:
        # Set the color value boundaries and step size
        if ((cb_upper != None) and (cb_lower != None)) and ((cb_upper != master_colorbar[param_idx][1][-1]) or (cb_lower != master_colorbar[param_idx][1][0])):
            bounds = np.linspace(cb_lower, cb_upper, num = int(len(master_colorbar[param_idx][1]))).tolist()
        else:
            bounds = master_colorbar[param_idx][1]
        step_size = bounds[1] - bounds[0]

        # Make the contour trace
        dat1 = go.Contour(
            # Get the data matrix for the contour using the current parameter and lateral index
            #   from the y slider
            # Transpose the matrix
            z = np.transpose(master_vargrid[param_idx][:, y_idx, :]),
            # Define the horizontal axis starting point and spacing (longitudinal slice)
            dx = abs(xgrid_vec[1] - xgrid_vec[0]),
            x0 = xgrid_vec[0],
            # Define the vertical axis starting point and spacing (depth)
            dy = (-1*abs(zgrid_vec[1] - zgrid_vec[0])),
            y0 = zgrid_vec[0],
            # Set the hover data name
            name = '<br><br>'+ master_label[param_idx],
            hoverinfo = 'x+y+z+name',
            # Turn off autocontouring and define the levels below
            autocontour = False,
            contours = dict(
                # Set to type levels (constraints can't handle multiple arbitrary levels)
                #   Limitation of levels is it must be equal spacing
                type = 'levels',
                # Set the colorbar properties from the master_colorbar given the current parameter
                # Set the lower value
                start = bounds[0],
                # Set the upper value
                end = bounds[-1],
                # Set the spacing
                size = step_size,
                # Remove the dividing lines on the colorbar
                showlines = False
            ),
            # Turn off the autocolorscale and define the colors to use below
            autocolorscale = False,
            # Set the colors corresponding to the previously defined levels
            colorscale = master_colorbar[param_idx][3],
            # Show the colorbar
            showscale = False
        )

        # Make the scatter trace for the bed
        dat2 = go.Scatter(
            # Assign the bed depth along the river longitudinally
            x = xgrid_vec,
            y = bed[:, y_idx],
            # Set the hover name as Depth with the actual value
            name = 'Bathymetry',
            hoverinfo = 'y+name',
            # Use a line and marker for the bed depth
            mode = 'lines+markers',
            # Marker properties
            marker = dict(
                size = 5,
                color = 'rgba(0, 0, 0, .8)'
            ),
            # Line properties
            line = dict(
                width = 2,
                color = 'rgba(0, 0, 0, .8)'
            )
        )

        # Return the figure with the contour trace and bed depth trace together
        return go.Figure(data = [dat1, dat2], layout = lay_2dy)

    else:
        raise dashNoUpdate

# Updates Overhead Figure when the z slider changes
@app.callback(
    dashOut('2dz', 'figure'),
    [dashIn('slider-z', 'value'), dashIn('main-param', 'value'), dashIn('cbar', 'src')],
    state = [dashState('cbar-ubound', 'value'), dashState('cbar-lbound', 'value')]
)
def update_zfig(z_idx, param_idx, cbimg, cb_upper, cb_lower):
    if cbimg != None:
        # Set the color value boundaries and step size
        if ((cb_upper != None) and (cb_lower != None)) and ((cb_upper != master_colorbar[param_idx][1][-1]) or (cb_lower != master_colorbar[param_idx][1][0])):
            bounds = np.linspace(cb_lower, cb_upper, num = int(len(master_colorbar[param_idx][1]))).tolist()
        else:
            bounds = master_colorbar[param_idx][1]
        step_size = bounds[1] - bounds[0]

        # Make the contour trace
        fig_data = go.Contour(
            # Get the data matrix for the contour using the current parameter and vertical index
            #   from the z slider
            # Transpose the matrix
            z = np.transpose(master_vargrid[param_idx][:, :, z_idx]),
            # Define the horizontal axis starting point and spacing (longitudinal slice)
            dx = abs(xgrid_vec[1] - xgrid_vec[0]),
            x0 = xgrid_vec[0],
            # Define the vertical axis starting point and spacing (cross-section)
            dy = abs(ygrid_vec[1] - ygrid_vec[0]),
            y0 = ygrid_vec[0],
            # Set the hover data name
            name = '<br><br>'+ master_label[param_idx],
            hoverinfo = 'x+y+z+name',
            # Turn off autocontouring and define the levels below
            autocontour = False,
            contours = dict(
                # Set to type levels (constraints can't handle multiple arbitrary levels)
                #   Limitation of levels is it must be equal spacing
                type = 'levels',
                # Set the colorbar properties from the master_colorbar given the current parameter
                # Set the lower value
                start = bounds[0],
                # Set the upper value
                end = bounds[-1],
                # Set the spacing
                size = step_size,
                # Remove the dividing lines on the colorbar
                showlines = False
            ),
            # Turn off the autocolorscale and define the colors to use below
            autocolorscale = False,
            # Set the colors corresponding to the previously defined levels
            colorscale = master_colorbar[param_idx][3],
            # Show the colorbar
            showscale = False
        )

        # Return the figure with the contour trace
        return go.Figure(data = [fig_data], layout = lay_2dz)

    else:
        raise dashNoUpdate

## Below are the callbacks to update the dynamic sliders (x and z) ##
# Dash cannot support multiple outputs yet so they are broken up into three updates for
#   the min, max and marks

# Updates the min value for the x slider
@app.callback(
    dashOut('slider-x', 'min'),
    [dashIn('slider-x', 'value')]
)
def update_slider_x_min(x_idx):
    # If the index value is below the (display width/2) + 1 then...
    if x_idx < (swid/2):
        # Set the lower indicies for the slider
        li = int(0)

    elif x_idx >= (swid/2) and x_idx < (slim_x - (swid/2)):
        # Set the lower indicies for the slider
        li = int(x_idx - (swid/2) + 1)

    elif x_idx >= (slim_x - (swid/2)):
        # Set the lower indicies for the slider
        li = int(slim_x - swid + 1)

    # Return the lower index
    return li

# Updates the max value for the x slider
@app.callback(
    dashOut('slider-x', 'max'),
    [dashIn('slider-x', 'value')]
)
def update_slider_x_max(x_idx):
    # If the index value is below the (display width/2) + 1 then...
    if x_idx < (swid/2):
        # Set the upper indicies for the slider
        ui = int(swid - 1)

    elif x_idx >= (swid/2) and x_idx < (slim_x - (swid/2)):
        # Set the upper  indicies for the slider
        ui = int(x_idx + (swid/2))

    elif x_idx >= (slim_x - (swid/2)):
        # Set the upper  indicies for the slider
        ui = int(slim_x)
    
    # Return the upper index
    return ui

# Updates the marks dictionary for the x slider with the new min and max
@app.callback(
    dashOut('slider-x', 'marks'),
    [dashIn('slider-x', 'min'), dashIn('slider-x', 'max')]
)
def update_slider_x_marks(li, ui):
    # Return the "filtered" xdict using the lower and upper indicies from the x slider
    return dict(list(xdict.items())[li:ui])

# Updates the min value for the y slider
@app.callback(
    dashOut('slider-y', 'min'),
    [dashIn('slider-y', 'value')]
)
def update_slider_y_min(y_idx):
    # If the index value is below the (display width/2) + 1 then...
    if y_idx < (swid/2):
        # Set the lower indicies for the slider
        li = int(0)

    elif y_idx >= (swid/2) and y_idx < (slim_y - (swid/2)):
        # Set the lower indicies for the slider
        li = int(y_idx - (swid/2) + 1)

    elif y_idx >= (slim_y - (swid/2)):
        # Set the lower indicies for the slider
        li = int(slim_y - swid + 1)

    # Return the lower index
    return li

# Updates the max value for the z slider
@app.callback(
    dashOut('slider-y', 'max'),
    [dashIn('slider-y', 'value')]
)
def update_slider_y_max(y_idx):
    # If the index value is below the (display width/2) + 1 then...
    if y_idx < (swid/2):
        # Set the upper indicies for the slider
        ui = int(swid - 1)

    elif y_idx >= (swid/2) and y_idx < (slim_y - (swid/2)):
        # Set the upper  indicies for the slider
        ui = int(y_idx + (swid/2))

    elif y_idx >= (slim_y - (swid/2)):
        # Set the upper  indicies for the slider
        ui = int(slim_y)
    
    # Return the upper index
    return ui

# Updates the marks dictionary for the z slider with the new min and max
@app.callback(
    dashOut('slider-y', 'marks'),
    [dashIn('slider-y', 'min'), dashIn('slider-y', 'max')]
)
def update_slider_y_marks(li, ui):
    # Return the "filtered" zdict using the lower and upper indicies from the z slider
    return dict(list(ydict.items())[li:ui])

# Updates the min value for the z slider
@app.callback(
    dashOut('slider-z', 'min'),
    [dashIn('slider-z', 'value')]
)
def update_slider_z_min(z_idx):
    # If the index value is below the (display width/2) + 1 then...
    if z_idx < (swid/2):
        # Set the lower indicies for the slider
        li = int(0)

    elif z_idx >= (swid/2) and z_idx < (slim_z - (swid/2)):
        # Set the lower indicies for the slider
        li = int(z_idx - (swid/2) + 1)

    elif z_idx >= (slim_z - (swid/2)):
        # Set the lower indicies for the slider
        li = int(slim_z - swid + 1)

    # Return the lower index
    return li

# Updates the max value for the z slider
@app.callback(
    dashOut('slider-z', 'max'),
    [dashIn('slider-z', 'value')]
)
def update_slider_z_max(z_idx):
    # If the index value is below the (display width/2) + 1 then...
    if z_idx < (swid/2):
        # Set the upper indicies for the slider
        ui = int(swid - 1)

    elif z_idx >= (swid/2) and z_idx < (slim_z - (swid/2)):
        # Set the upper  indicies for the slider
        ui = int(z_idx + (swid/2))

    elif z_idx >= (slim_z - (swid/2)):
        # Set the upper  indicies for the slider
        ui = int(slim_z)
    
    # Return the upper index
    return ui

# Updates the marks dictionary for the z slider with the new min and max
@app.callback(
    dashOut('slider-z', 'marks'),
    [dashIn('slider-z', 'min'), dashIn('slider-z', 'max')]
)
def update_slider_z_marks(li, ui):
    # Return the "filtered" zdict using the lower and upper indicies from the z slider
    return dict(list(zdict.items())[li:ui])

# Fire the dashboard up
app.run_server(debug = False)
