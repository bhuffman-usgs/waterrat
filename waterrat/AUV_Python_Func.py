# Import all necessary libraries and functions
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os, sys, pyproj, utm, re, inspect
import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showerror
import plotly as plty
import plotly.graph_objs as go
import numpy as np
from numpy import multiply as mm, divide as md, sqrt as msqrt, square as ms, power as mpow
from io import BytesIO
import base64
from datetime import datetime
from math import floor
from random import sample as randomSample
from cefpython3 import cefpython as cef
import ctypes

class processDataframe(object):

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def columnDrop(self, keys):
        if type(keys) == list:
            for key in keys:
                try:
                    self.dataframe.drop(key, axis = 1, inplace = True)
                except KeyError:
                    pass
        else:
            print('Failed at line %i.' % inspect.currentframe().f_back.f_lineno)
            print('Variable keys must be in a list.')
            sys.exit(0)

    def columnDropStartsWith(self, keys):
        if type(keys) == list:
            [self.dataframe.drop(column, axis = 1, inplace = True) for k in keys for column in self.dataframe if column.startswith(k)]
        else:
            print('Failed at line %i.' % inspect.currentframe().f_back.f_lineno)
            print('Variable keys must be in a list.')
            sys.exit(0)

    def columnTryRename(self, keys, newKeys):
        if (type(keys) == list) and (type(newKeys) == list):
            for k, nk in zip(keys, newKeys):
                try:
                    _ = self.dataframe[k]
                    self.dataframe.rename(columns = {k: nk}, inplace = True)
                except KeyError:
                    pass
        else:
            print('Failed at line %i.' % inspect.currentframe().f_back.f_lineno)
            print('Variables keys and newKeys must be lists.')
            sys.exit(0)

    def columnTryConvert(self, keys, convKeys, conversions):
        if (type(keys) == list) and (type(convKeys) == list) and (type(conversions) == list):
            for k, ck, c in zip(keys, convKeys, conversions):
                try:
                    _ = self.dataframe[k].values
                except KeyError:
                    try:
                        self.dataframe[k] = self.dataframe[ck].values*c
                        del self.dataframe[ck]
                    except KeyError:
                        print('Failed at line %i.' % inspect.currentframe().f_back.f_lineno)
                        print(''.join(["No data for columns '", k, "' and '", ck, "'."]))
        else:
            print('Failed at line %i.' % inspect.currentframe().f_back.f_lineno)
            print('Variables keys, newKeys and converions must be lists.')
            sys.exit(0)

    def columnMerge(self, keyOne, seperator, keyTwo):
        self.dataframe[' '.join([keyOne, keyTwo])] = self.dataframe[keyOne] + seperator + self.dataframe[keyTwo]

    def secondsFromEpoch(self, key, keyFormat):
        # Get the date time data as a numpy array
        date = self.dataframe[key].values
        # Convert the first date string into seconds since 1/1/1970 00:00:00
        date[0] = (datetime.strptime(date[0], keyFormat) - datetime(1970, 1, 1)).total_seconds() 
        # Loop over the remaining date strings
        for i in range(1, len(date), 1):
            # Convert the date string into seconds 1/1/1970 00:00:00 and make relative to the first logged time
            date[i] = (datetime.strptime(date[i], keyFormat) - datetime(1970, 1, 1)).total_seconds() - date[0]
        # Reassign the date time column as the seconds since 1/1/1970 00:00:00
        self.dataframe[key] = date.astype(float)

    def coordinateCorrection(self):
        # Store the uncorrected latitude and longitude in the dataframe
        self.dataframe[' '.join(['Latitude', 'Uncorrected'])] = self.dataframe['Latitude'].values
        self.dataframe[' '.join(['Longitude', 'Uncorrected'])] = self.dataframe['Longitude'].values
        # Get the lat/lon and number of satellites data as numpy arrays
        lat = self.dataframe['Latitude'].values
        lon = self.dataframe['Longitude'].values
        nSatellite = self.dataframe['Number of Sats'].values
        # Extract the UTM zone from the coordinates
        zone = utm.from_latlon(lat[0], lon[0])
        zone = str(zone[2]) + zone[3]
        # Generate a projection to convert lat/lon to easting/northing (meters) based on the UTM zone and the WGS84 ellipsoid
        p = pyproj.Proj(proj = 'utm', zone = zone, ellps = 'WGS84')
        # Apply the projection
        x, y = p(lon, lat)
        # Get the distances between consecutive (x, y) points
        trackDiff = euc_dist(x, y)
        # Get the change in satellites that the AUV is currently locked onto
        satelliteDiff = np.append(np.zeros(10), (np.asarray(nSatellite[10:]) - np.asarray(nSatellite[:-10])))
        # Get indices where the AUV traveled further than 3 meters between points AND when there was a positive jump in locked satellites
        #   This indicates that the AUV missed its target waypoint when it surfaced
        booIndex = mm((trackDiff >= 3), (satelliteDiff != 0))
        driftIndex = np.where(booIndex == True)[0]
        # Clear memory
        del zone, trackDiff, satelliteDiff, booIndex
        # If there is no indication of drifting notify the user
        if len(driftIndex) == 0:
            print("No drifting was found to occur in the survey.")
        # If drifting was found to occur...
        else:
            # Get the current step data as a numpy array
            currentStep = self.dataframe['Current Step'].values
            # Make the dive step and index variable list where the dive step is equal to the step before driting occured
            #   and the dive index are the indices where the dive step occured at in the data log
            diveList = [[(currentStep[index] - 1), np.where(currentStep == (currentStep[index] - 1))[0]] for index in driftIndex]
            # Enumerate the possible dive indicies
            for i, diveItem in enumerate(diveList):
                # Unpack the dive step and index from the dive list
                step = diveItem[0]
                index = diveItem[1]
                # Loop while the number satellites is equal to zero (the AUV is under water)
                while nSatellite[index[0]] == 0:
                    # Move the step back one
                    step -= 1
                    # Again find the indices where this step occured at in the data log
                    index = np.where(currentStep == step)[0]
                # Overwrite the current diveList item with the first indice in the dive index list
                diveList[i] = index[0]
            # Convert the list to a integer type numpy array and rename the variable
            diveIndex = np.asarray(diveList, dtype = int)
            # Clear memory
            del nSatellite, currentStep, diveList, step
            # Generate the WGS84 ellipsoid
            wgs = pyproj.Geod(ellps = 'WGS84')
            # Copy the longitude and latitude data to a numpy array
            latCorrected = np.copy(lat)
            lonCorrected = np.copy(lon)
            # Loop over the dive and drift indices
            for i, j in zip(diveIndex, driftIndex):
                # Determine the AUV's heading and distance to, between the dive and drift location
                diveHeading, _, diveDiff = wgs.inv(lon[i], lat[i], lon[j], lat[j])
                # Make an index vector that ranges from the dive to drift indices
                index = np.arange((i + 1), j, 1)
                # Discretize the distance from the dive to drift location
                dDiff = (diveDiff/len(index))
                # Initialize the total distance variable
                tDist = dDiff
                # Loop over the index vector
                for ii in index:
                    # Use the lat/lon corresponding to the dive index AND the dive heading and distance from the dive to correct the lat/lon coordinates
                    lonCorrected[ii], latCorrected[ii], _ = wgs.fwd(lon[i], lat[i], diveHeading, tDist)
                    # Go to the next location along the drifted track
                    tDist += dDiff
            # Store the corrected lat/lon coordinates
            self.dataframe['Latitude'] = latCorrected
            self.dataframe['Longitude'] = lonCorrected
            # Store the corrected lat/lon coordinates as UTM coordinates in feet
            x, y = p(lonCorrected, latCorrected)
        self.dataframe['X (ft)'] = x*3.28084
        self.dataframe['Y (ft)'] = y*3.28084

    def depthCorrection(self, offset):
        # Add an offset to the water column and dfs depth columns in the dataframe
        self.dataframe['Total Water Column (ft)'] = self.dataframe['Total Water Column (ft)'].values + offset
        self.dataframe['DFS Depth (ft)'] = self.dataframe['DFS Depth (ft)'].values + (offset/2)

    def distanceCompute(self):
        # Get the X and Y coordinate data as numpy arrays
        x = self.dataframe['X (ft)'].values
        y = self.dataframe['Y (ft)'].values
        # Get the current step data as a numpy array and add one step to get a destination step
        destinationStep = self.dataframe['Current Step'].values + 1
        # Get the total number of waypoints in the log
        nWaypoints = max(destinationStep)
        # Initialize the distance array
        diffDist = np.empty((0,))
        # Loop over each waypoint step
        for i in np.arange(1, (nWaypoints + 1), 1):
            # Get the indices where the current waypoint step matches
            ii = np.where(destinationStep == i)[0]
            # Get the cummulative distance from the current waypoint
            diffDist = np.append(diffDist, np.cumsum(euc_dist(x[ii], y[ii])))
        # Store the differential distances between points, the distance travel since the start
        #   and the distance traveled between waypoints
        self.dataframe['Distance Between (ft)'] = euc_dist(x, y)
        self.dataframe['Track Distance (ft)'] = np.cumsum(euc_dist(x, y))
        self.dataframe['Distance From Waypoint (ft)'] = diffDist      

    def pressureSalinityDensityCompute(self):
        try:
            self.dataframe['Pressure (dbar)'], self.dataframe['Salinity (ppt)'] = comp_press_sal(self.dataframe['SpCond uS/cm'].values/1000, self.dataframe['Temp C'].values, self.dataframe['DFS Depth (ft)'].values/3.28084, self.dataframe['Latitude'].values)
        except KeyError:
            self.dataframe['SpCond uS/cm'] = np.zeros(len(self.dataframe[0]))
            self.dataframe['Pressure (dbar)'], self.dataframe['Salinity (ppt)'] = comp_press_sal(self.dataframe['SpCond uS/cm'].values, self.dataframe['Temp C'].values, self.dataframe['DFS Depth (ft)'].values/3.28084, self.dataframe['Latitude'].values)
            print('No specific conductance data present, values set to 0 for computing pressure, salinity and density.')
        self.dataframe['Density (kg/m3)'] = comp_dens(self.dataframe['Salinity (ppt)'].values, self.dataframe['Temp C'].values, self.dataframe['Pressure (dbar)'].values)
        
    def graphRawData(self, figureList, labelList, figureOption):
        if not hasattr(self, 'matPlotLib'):
            self.matPlotLib = plt
        if not hasattr(self, 'figureNumber'):
            self.figureNumber = 1
        for plot, label in zip(figureList, labelList):
            fig = self.matPlotLib.figure(self.figureNumber)
            if figureOption == 0:
                subLocation = len(plot)*100 + 11
                passed = False
                for subPlot in plot:
                    try:
                        if not passed:
                            ax = fig.add_subplot(subLocation)
                        ax.plot(self.dataframe[subPlot[0]].values*subPlot[2], subPlot[3])
                        ax.yaxis.set_label_text(subPlot[1])
                        subLocation += 1
                        passed = False
                    except KeyError:
                        passed = True
                        pass
                ax.xaxis.set_label_text(label)
            elif figureOption == 1:
                ax = fig.add_subplot(111)
                for subPlot in plot:
                    try:
                        ax.plot(self.dataframe[subPlot[0]].values*subPlot[2], subPlot[3], label = subPlot[1])
                    except KeyError:
                        pass
                ax.xaxis.set_label_text(label[0])
                ax.yaxis.set_label_text(label[1])
                ax.legend()
            self.figureNumber += 1
    
    # Method to run the robust lowess filtering algorithm on the selected keys in the dataframe used
    #   each window size in the input window size range.  When the algorithm is done, a interactive
    #   gui is displayed to the user to select which window size filter is appropriate for each key
    #   in the dataframe.
    def graphSmoothData(self, yKeys, windowSizes):
        # Make an ordered list of window sizes to run the robust lowess filter
        windows = np.arange(windowSizes[0], windowSizes[1] + 0.1, 1)
        # Initialize the filtered data list and an array to identify if a key was used
        yFiltered = []
        keyUsed = np.zeros(len(yKeys), dtype = np.bool)
        # Loop over the keys (parameters)
        for i, yKey in enumerate(yKeys):
            # Attempt to filter this parameter
            try:
                # Initialize the temporary filtered data list
                yFilteredTemp = []
                # For each window size run the robust lowess filter and store the
                #   filtered data to the temporary list
                for window in windows:
                    yf = robustLowess(self.dataframe[yKey].values, window)
                    yFilteredTemp.append(yf)
                # Store the filtered data for this parameter
                yFiltered.append(yFilteredTemp)
                # Key was found in the dataset
                keyUsed[i] = 1
            except KeyError:
                print(''.join(["Smoothing: No data column '", yKey, "' found."]))
                # Key was not found in the dataset
                keyUsed[i] = 0
        # Subset the yKeys list based on what data was present
        yKeys = np.asarray(yKeys)
        usedKeys = yKeys[keyUsed].tolist()
        # Build the tkinter html embedded frame (before CEF is started)
        h = 240 + 20*len(windows)
        if h > 1000: h = 1000
        smooth = smoothingFrame(createRoot('Smoothing Window Selection', 1400, h), self.dataframe, usedKeys, yFiltered, windows)
        # Start up CEF
        cef.Initialize()
        # Run the gui
        smooth.mainloop()
        # When the gui is closed, shut down CEF
        cef.Shutdown()
        # Check if the smoothing gui was exited properly
        if smooth.closedProperly:
            # If so then update the data with the chosen filtered data
            for key, data, choice in zip(usedKeys, yFiltered, smooth.choices):
                # Check if the data was chosen to be smoothed
                if choice != -1:
                    self.dataframe[key] = data[choice]
        else:
            print('Failed at line %i.' % inspect.currentframe().f_back.f_lineno)
            print('The smoothing GUI was closed improperly.\nSelect a smoothing option and click the "Done Smoothing" button when finished.')
            sys.exit(0)

######################### Class Definitions #########################
# This class creates an html embedded tkinter frame (plotly compatible)
#   inside the initial root along with functional radio buttons and
#   an next/exit button for selecting smoothing windows
class smoothingFrame(tk.Frame):
    def __init__(self, root, dataframe, keys, filtered, windows):
        # Lock the frame's size
        root.resizable(False, False)
        # Initialize this class object using root and tk.Frame.__init__ (gain its properties)
        tk.Frame.__init__(self, root)
        # Bind the windows close button with the onClose method
        self.master.protocol("WM_DELETE_WINDOW", self.onClose)
        # Define a variable to determine how the frame was closed
        self.closedProperly = False
        # Define the choice storage 
        self.choices = np.full(len(keys), -1, dtype = int)
        # Set a global variable from the radio buttons to assign their values to
        self.var = tk.IntVar()
        # Loop over the window sizes and create a radio button, linking it with self.var
        opt = tk.Radiobutton(self, text = 'No Smoothing', variable = self.var, value = -1)
        opt.grid(row = 1, column = 1)
        for i, window in enumerate(windows):
            opt = tk.Radiobutton(self, text = ''.join(['n = ', str(int(window))]), variable = self.var, value = i)
            opt.grid(row = i + 2, column = 1)
        # Create a button to go to the next parameter or exit the smoothing frame
        tk.Button(self, text = "Done Smoothing", command = self.onDone).grid(row = len(windows) + 2, column = 1, padx = 10)
        # Initialize the html embedded frame using the html file path
        self.plotFrame = self.browserFrame(self, dataframe, keys, filtered, windows)
        # Finalize the frame layout
        self.plotFrame.grid(row = 0, column = 0, rowspan = len(windows) + 4, sticky = tk.NSEW, ipadx = 10, padx = 10, pady = 10)
        self.grid_rowconfigure(0, weight = 1)
        self.grid_rowconfigure(len(windows) + 3, weight = 1)
        self.grid_columnconfigure(0, weight = 1)
        self.grid_columnconfigure(1, weight = 0)
        self.pack(fill = tk.BOTH, expand = tk.YES)

    # Method to cleanly close the gui 
    def onClose(self):
        if self.plotFrame.control:
            if self.plotFrame:
                if self.plotFrame.browser:
                    self.plotFrame.browser.CloseBrowser(True)
                    self.plotFrame.browser = None
                self.plotFrame.destroy()
            self.master.destroy()

    # Method to store the chosen filter data index and go to
    #   the next paramter index.  If there are no parameters left
    #   close the gui
    def onDone(self):
        self.choices[self.plotFrame.paramIndex] = self.var.get()
        if (self.plotFrame.paramIndex + 1) == len(self.plotFrame.keys):
            self.closedProperly = True
            self.onClose()
        else:
            self.plotFrame.paramIndex += 1
            self.plotFrame.control = False

    # This subclass creates the html embedded tkinter frame and populates it
    #   within the main frame
    class browserFrame(tk.Frame):
        def __init__(self, root, dataframe, keys, filtered, windows):
            # Initialize the cef loop controller
            self.control = False
            # Store the dataframe, keys, filtered data, windows and set the
            #   current parameter index
            self.dataframe = dataframe
            self.keys = keys
            self.filtered = filtered
            self.windows = windows
            self.paramIndex = 0
            # Create the plotly graph for the raw data and all the associated filtered data
            self.buildPlot()
            # Set the path to the plotly graph
            self.htmlFile = ''.join([os.getcwd(), '\\multi-smooth-plot.html'])
            # Initialize this subclass object using root and tk.Frame.__init__ (gain its properties)
            tk.Frame.__init__(self, root)
            # Bind the loading of the html frame with the configureBrowser method
            self.bind("<Configure>", self.configureBrowser)

        # Method to build the CEF browser within the html frame
        def configureBrowser(self, _):                 
            # Set up the cef browser id and size
            winfo = cef.WindowInfo()
            winfo.SetAsChild(self.winfo_id(), [0, 0, self.winfo_width(), self.winfo_height()])
            # Link the browser to the plotly graph
            self.browser = cef.CreateBrowserSync(winfo, url = self.htmlFile)
            # Allow the cef loop to run and start it
            self.control = True
            self.handleMessages()

        # Method to construct the plotly graph for the raw data and associated filtered data
        def buildPlot(self):
            # Create an ordered index list
            x = np.arange(0, len(self.dataframe.index), 1)
            # Initialize the figure data list
            figureData = []
            # Add the raw data scatter point data to the figure
            figureData.append(go.Scatter(
                x = x, y = self.dataframe[self.keys[self.paramIndex]].values,
                name = 'Raw Data', mode = 'markers', 
                marker = dict(color = 'rgb(255, 0, 0)', size = 5)
            ))
            # Add the filtered data (by window size) to the figure
            for d, window in zip(self.filtered[self.paramIndex], self.windows):
                figureData.append(go.Scatter(
                    x = x, y = d,
                    name = ''.join(['n = ', str(int(window))]), mode = 'lines', 
                    line = dict(color = randomColorHex(), width = 3)
                ))
            # Modify the figure layout
            figureLayout = go.Layout(
                font = dict(family = 'PT Sans Narrow', size = 12),
                xaxis = dict(title = 'Sample Number', range = [-1, len(self.dataframe.index)], linecolor = 'rgb(0, 0, 0)', linewidth = 1, mirror = True),
                yaxis = dict(title = self.keys[self.paramIndex], linecolor = 'rgb(0, 0, 0)', linewidth = 1, mirror = True)
            )
            # Create the figure and save it to file
            figure = go.Figure(data = figureData, layout = figureLayout)
            plty.offline.plot(figure, filename = 'multi-smooth-plot.html', auto_open = False)

        # Method to run the message/event loop for CEF
        def handleMessages(self):
            cef.MessageLoopWork()
            if self.control:
                # Continue the message/event loop
                self.after(10, self.handleMessages)
            else:
                # User has hit the done smoothing button and the next parameter
                #   is to be viewed.  Rebuild the plotly graph and load it to the
                #   browser.  Continue the message/event loop
                self.buildPlot()
                self.browser.LoadUrl(self.htmlFile)
                self.control = True
                self.after(10, self.handleMessages)
        
# This class allows a user to replace sub-string(s) in the original string based on regular expression pattern(s)
#   Requires the following group names in the regex pattern(s): prefix, keep, edit, and suffix
#     where the edit group will be replaced in the replace method
class stringSubReplace(object):
    # Init method to define original and modified strings
    def __init__(self, string):
        # Store the string as an old and new instance variable
        self.original = string
        # Set the new string method equal to the old string method
        self.modified = string
    
    # Method to replacem sub-string(s) based on pattern(s) matched to the original string
    def replace(self, pattern, replacement):
        # Compile the regular expression pattern(s)
        compPattern = [re.compile(p) for p in pattern]
        # Loop over the patterns
        for p, r in zip(compPattern, replacement):
            # Using each pattern replace them with the 
            match = p.match(self.original)
            self.modified = match.group('prefix') + match.group('keep') + r + match.group('suffix')

# This class creates a file dialog window for the user to select a data file
class fDialog(object):
    # Init method to define the file dialog object
    def __init__(self, title, startDir, fileType, *fileFlag):
        # Create a custom tkinter window object and assign it as master
        self.master = createRoot(' '.join([title, fileType[0]]), 400, 50)
        # Link a button to the master with text 'Browse to File', linked to the loadFile method, with a width of 10
        self.button = tk.Button(self.master, text = 'Browse to File', command = self.loadFile, width = 10)
        # Pack the button (packing helps condense the visual layout)
        self.button.pack()
        # Assign the starting directory and file type
        self.startDir = startDir
        self.fileType = fileType
        # Set a file text flagger incase the user does not select a defined file
        if fileFlag:
            self.fileFlag = fileFlag[0]
        else:
            self.fileFlag = ''

    # Method to select a data file when the 'Browse to File' button is clicked
    def loadFile(self):
        # Open a file selection prompt limiting by file type and using the starting directory
        fname = askopenfilename(initialdir = self.startDir, filetypes = [self.fileType])
        # If the file name exists
        if fname:
            try:
                if self.fileFlag == '':
                    # Set the .filename property in the class to the selected filepath
                    self.filename = fname
                else:
                    # Make sure the ending of the file is matches the flag
                    if fname[-(len(self.fileFlag) + 4):-4] == self.fileFlag:
                        # Set the .filename property in the class to the selected filepath
                        self.filename = fname
                    else:
                        # Prompt user the wrong file was selected
                        showerror('Source File', ''.join(["Select a file with '", self.fileFlag, "' appended to the name."]))
                        # Stop execution
                        sys.exit(0)
            except:
                # Prompt user that file could not be selected
                showerror('Source File', "Failed to read file\n'%s'" % fname)
                # Stop execution
                sys.exit(0)

            # Destroy/close the master frame
            self.master.destroy()

# Create a case load class object
class case_load(object):
    # Define the main method to load a specific data type, the colormap for it, and the value bounds corresponding to each color in the colormap
    def get_dfvar(self, arg1, arg2):
        # Use the dictionary function to grab the number corresponding to the data type, then construct a string labeled as "load_[number]"
        dfvar = 'load_' + str(dict_load(arg1))
        # Run the sub method that corresponds to the "load_[number]"
        method = getattr(self, dfvar, lambda: "No such data variable")
        # Return the variables from the sub method
        return method(arg2)

    # Each sub-method is constructed the same with a structure to pull data from the dataframe for the specific data type, to generate a colormap of rgb vectors and a bounds matrix that contains the limits for each rgb vector in the colormap
    def load_1(self, arg1):
        # Load data (here 1 corresponds to temperature)
        var = np.asarray(arg1['Temp C'])
        # Make an official label string
        label = 'Temperature'
        # Make a unit label
        unit = 'Degrees Celsius'
        # Make a short hand label
        unit_short = '&deg;C'
        # Load the colormap
        map = np.asarray([[0, 0, 0.5625], [0, 0, 0.8750], [0, 0.1875, 1.0000], [0, 0.5000, 1.0000], [0, 0.8125, 1.0000], [0.1250, 1.0000, 0.8750], [0.4375, 1.0000, 0.5625], [0.7500, 1.0000, 0.2500], [1.0000, 0.9375, 0], [1.0000, 0.6250, 0], [1.0000, 0.3125, 0], [1.0000, 0, 0], [0.6875, 0, 0]])
        # Compositional color
        comp_map = [143, 0, 112]
        # Load the bounds
        bounds = [0, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
        # Load the sensor resolution
        res = [0.2]
        # Return the data, colormap and bounds
        return var, label, unit, unit_short, map, comp_map, bounds, res
    # The following sub methods share the same structure as above but with different data types, colormaps and bounds to be loaded;  Thus they will not be commented as to how they work
    def load_2(self, arg1):
        var = np.asarray(arg1['SpCond uS/cm'])
        label = 'Specific Conductance'
        unit = 'Microsiemens per Centimeter'
        unit_short = '&mu;S/cm'
        map = np.asarray([[0.4000, 0, 0.2000], [0.4786, 0, 0.2333], [0.5571, 0, 0.2667], [0.6357, 0, 0.3000], [0.7143, 0, 0.3333], [0.7929, 0, 0.3667], [0.8406, 0.0914, 0.4101], [0.8781, 0.2132, 0.4569], [0.9156, 0.3350, 0.5038], [0.9531, 0.4568, 0.5506], [0.9906, 0.5786, 0.5974], [1.0000, 0.6351, 0.6351], [1.0000, 0.6698, 0.6698], [1.0000, 0.7045, 0.7045], [1.0000, 0.7393, 0.7393], [1.0000, 0.7740, 0.7740]])
        comp_map = [22, 170, 127]
        bounds = [0, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160]
        res = [1]
        return var, label, unit, unit_short, map, comp_map, bounds, res
    def load_3(self, arg1):
        var = np.asarray(arg1['pH'])
        label = 'pH'
        unit = 'pH'
        unit_short = ''
        map = np.asarray([[0, .4, .6], [0, .5, .7], [0, .6, .8], [0, .7, .9], [0, .8, 1], [.2667, .9333, .8667], [.4, 1, .8], [.2902, .8902, .6902], [.0706, .6706, .4706], [0, .6, .45], [0, .5216, .4], [0, .45, .4], [0, .4, .4], [0, .2, .2]])
        comp_map = [181, 28, 79]
        bounds = [0, 5.5, 5.7, 5.9, 6.1, 6.3, 6.5, 6.7, 6.9, 7.1, 7.3, 7.5, 7.7, 7.9]
        res = [0.1]
        return var, label, unit, unit_short, map, comp_map, bounds, res
    def load_4(self, arg1):
        var = np.asarray(arg1['Turbid NTU'])
        label = 'Turbidity'
        unit = 'Nephelometric Turbidity Units'
        unit_short = 'NTU'
        map = np.asarray([[0.8392, 0.7216, 0.5882], [0.7333, 0.6000, 0.4431], [0.6235, 0.4784, 0.2980], [0.5451, 0.3961, 0.2118], [0.4706, 0.3255, 0.1451], [0.3961, 0.2588, 0.0784], [0.3137, 0.1804, 0.0039]])
        comp_map = [116, 154, 201]
        bounds = [0, 5, 10, 15, 20, 25, 30]
        res = [1]
        return var, label, unit, unit_short, map, comp_map, bounds, res
    def load_5(self, arg1):
        var = np.asarray(arg1['ODO mg/L'])
        label = 'Dissolved Oxygen'
        unit = 'Milligrams per Liter'
        unit_short = 'mg/L'
        map = np.asarray([[.3, 0, .3], [0.4522, 0, 0.4000], [0.5565, 0, 0.4000], [0.6609, 0, 0.4000], [0.7652, 0, 0.4000], [0.8500, 0.1000, 0.5500], [0.9250, 0.2500, 0.7750], [1.0000, 0.4000, 1.0000], [1.0000, 0.5000, 1.0000], [1.0000, 0.6000, 1.0000], [1.0000, 0.7000, 1.0000], [1.0000, 0.8000, 1.0000]])
        comp_map = [19, 191, 58]
        bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        res = [0.1]
        return var, label, unit, unit_short, map, comp_map, bounds, res
    def load_6(self, arg1):
        var = np.asarray(arg1['Chl ug/L'])
        label = 'Chlorophyll'
        unit = 'Micrograms per Liter'
        unit_short = '&mu;g/L'
        map = np.asarray([[0, .2824, 0], [0, .3765, 0], [0, .4902, 0], [0, .6157, 0], [0, .7373, 0], [0, .8275, 0], [0, .9412, 0], [0, 1, 0], [.6, 1, .6]])
        comp_map = [255, 67, 255]
        bounds = [0, 5, 10, 15, 20, 25, 30, 35, 40]
        res = [1]
        return var, label, unit, unit_short, map, comp_map, bounds, res
    def load_7(self, arg1):
        var = np.asarray(arg1['BGA-PC cells/mL'])
        label = 'Cyanobacteria'
        unit = 'Cells per Milliliter'
        unit_short = 'cells/mL'
        map = np.asarray([[0, 0, .549], [0, .294, .549], [0, .392, .588], [0, .588, .784], [0, .588, .686], [0, .627, .588], [0, .588, .353], [0, .686, .353], [.392, .784, 0], [.706, .882, .111], [.882, .98, .294], [.98, .98, .627]])
        comp_map = [255, 105, 165]
        bounds = [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000]
        res = [500]
        return var, label, unit, unit_short, map, comp_map, bounds, res
    def load_8(self, arg1):
        var = np.asarray(arg1['Rhodamine ug/L'])
        label = 'Rhodamine'
        unit = 'Micrograms per Liter'
        unit_short = '&mu;g/L'
        map = np.asarray([[.3, 0, .3], [0.4522, 0, 0.4000], [0.5565, 0, 0.4000], [0.6609, 0, 0.4000], [0.7652, 0, 0.4000], [0.8500, 0.1000, 0.5500], [0.9250, 0.2500, 0.7750], [1.0000, 0.4000, 1.0000], [1.0000, 0.5000, 1.0000], [1.0000, 0.6000, 1.0000], [1.0000, 0.7000, 1.0000], [1.0000, 0.8000, 1.0000]])
        comp_map = [19, 191, 58]
        bounds = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
        res = [1]
        return var, label, unit, unit_short, map, comp_map, bounds, res
    # def load_9(self, arg1):
    #     var = np.asarray(arg1['Salinity (ppt)'])
    #     label = 'Salinity'
    #     unit = 'Practical Salinity Units'
    #     unit_short = 'PSU'
    #     map = np.asarray([])
    #     comp_map = []
    #     bounds = []
    #     res = []
    #     return var, label, unit, unit_short, map, comp_map, bounds, res
    # def load_10(self, arg1):
    #     var = np.asarray(arg1['Density (kg/m3)'])
    #     label = 'Density'
    #     unit = 'Kilograms per Cubic Meter'
    #     unit_short = 'kg/m<sup>3</sup>'
    #     map = np.asarray([])
    #     comp_map = []
    #     bounds = []
    #     res = []
    #     return var, label, unit, unit_short, map, comp_map, bounds, res
#####################################################################

# Function/Dictionary for water quality data variables (variable name in --> number out)
def dict_load(argument):
    # Create the dictionaries keys and values
    switch_dict = {
        "Temp" : 1,
        "SpCond" : 2, 
        "pH" : 3,
        "Turb" : 4,
        "DO" : 5,
        "Chl" : 6,
        "BGA" : 7,
        "Rhod" : 8
        # "Sal" : 9,
        # "Dens" : 10
    }
    # Grab the value for the key using the input argument
    dict = switch_dict.get(argument)
    # Return the dictionary value
    return dict

######################### Function Definitions #########################
# Function to build a tkinter window object with a specific title and size
def createRoot(title, w, h):
    # Create a tkinter window object
    root = tk.Tk()
    # Get monitor size (pixels)
    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight()
    # Establish the windows top left corner location
    ws = (ws/2) - (w/2)
    hs = (hs/2) - (h/2)
    # Set the window's geometry
    root.geometry('%dx%d+%d+%d' % (w, h, ws, hs))
    # Set the window's title
    root.title(title)
    # Return the built tkinter window object
    return root

# Function to compute pressure [db] (from depth [m] and latitude) and salinity [PSU~ppt] (from pressure, specific conductance[mS/cm] and water temperature[degC])
def comp_press_sal(SpC, tempC, dep, lat):
    ### Calculate the pressure ###
    # Set up variable for radians
    rad = (np.pi/180) 
    # Convert latitude to radians
    xx = ms(np.sin(abs(lat)*rad))
    # Set constants
    AA = 1 - ((5.92 + (5.25*xx))*.001)
    BB = 2.21e-6
    # Compute the pressure based on the latitude and depth 
    press = (AA - msqrt((ms(AA) - (4*BB*dep))))/(2*BB)
    # Clear memory
    del rad, xx, AA, BB
    ##############################

    # Set the temperature coeff. based on the YSI sonde manual (pg. 217)
    TC = 0.0191
    # Inverted equation for Specific Conductance at 25 deg C based on the YSI sonde manual (pg.217)
    Cond = mm(SpC, ((TC*(tempC - 25)) + 1))
    # Reference conductivity for water at a temperature of 15 deg C, 0 decibar of pressure and salinity equal to 35 psu (practical salinity unit)
    c3515 = 42.914
    # Set the conductivity ratio
    c_r = (Cond/c3515)
    del TC, Cond, c3515

    ### Calculate salinity (with a low salinity correction) ###
    # Set constants and compute the conductivity ratio (temperature polynomial)
    c0 =  0.6766097
    c1 =  2.00564e-2
    c2 =  1.104259e-4
    c3 = -6.9698e-7
    c4 =  1.0031e-9
    rt = c0 + mm((c1 + mm((c2 + mm((c3 + (c4*tempC)), tempC)), tempC)), tempC)
    # Clear memory
    del c0, c1, c2, c3, c4

    # Set constants and compute the conductivity ratio (pressure polynomial)
    d1 =  3.426e-2
    d2 =  4.464e-4
    d3 =  4.215e-1
    d4 = -3.107e-3
    e1 =  2.070e-5
    e2 = -6.370e-10
    e3 =  3.989e-15
    rp = 1 + md(mm(press, (e1 + (e2*press) + (e3*ms(press)))), (1 + (d1*tempC) + (d2*ms(tempC)) + mm((d3 + (d4*tempC)), c_r)))
    # Clear memory
    del d1, d2, d3, d4, e1, e2, e3

    # Set the conductivity ratio
    r = md(c_r, mm(rp, rt))
    # Clear memory
    del c_r, rp, rt

    # Set constants and compute the salinity
    a0 =  0.0080
    a1 = -0.1692
    a2 =  25.3851
    a3 =  14.0941
    a4 = -7.0261
    a5 =  2.7081
    b0 =  0.0005
    b1 = -0.0056
    b2 = -0.0066
    b3 = -0.0375
    b4 =  0.0636
    b5 = -0.0144
    k = 0.0162
    rtx = msqrt(r)
    del_t = (tempC - 15)
    del_s = mm(md(del_t, (1 + (k*del_t))), (b0 + mm((b1 + mm((b2 + mm((b3 + mm((b4 + (b5*rtx)), rtx)), rtx)), rtx)), rtx)))
    S = a0 + mm((a1 + mm((a2 + mm((a3 + mm((a4 + (a5*rtx)), rtx)), rtx)), rtx)), rtx) + del_s
    # Clear memory
    del a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5, k, rtx, del_t, del_s

    # Set constants and correct for low salinity
    a0 = 0.008
    b0 = 0.0005
    x0 = 400*r
    y0 = 100*r
    f0 = md((tempC - 15), (1 + (0.0162*(tempC - 15))))
    S_corr = (a0/(1 + (1.5*x0) + ms(x0))) + md((b0*f0), (1 + msqrt(y0) + mm(y0, msqrt(y0))))
    S = S - S_corr
    # Clear memory
    del a0, b0, x0, y0, f0, S_corr, r
    ###########################################################

    # Pass back the pressure and salinity
    return press, S

# Function to compute density [kg/m3] from salinity[psu], temperature[degC] and pressure[dbar]
def comp_dens(sal, tempC, pres):
    ### Compute the density of sea water at the surface ###
    # Set constants and calculate density of pure water
    a0 =  999.842594
    a1 =  6.793952e-2
    a2 = -9.095290e-3
    a3 =  1.001685e-4
    a4 = -1.120083e-6
    a5 =  6.536332e-9
    dens_pw = a0 + mm((a1 + mm((a2 + mm((a3 + mm((a4 + (a5*tempC)), tempC)), tempC)), tempC)), tempC)
    # Clear memory
    del a0, a1, a2, a3, a4, a5

    # Set constants and calculate density of sea water at atmospheric pressure
    b0 =  8.24493e-1
    b1 = -4.0899e-3
    b2 =  7.6438e-5
    b3 = -8.2467e-7
    b4 =  5.3875e-9
    c0 = -5.72466e-3
    c1 =  1.0227e-4
    c2 = -1.6546e-6
    d0 =  4.8314e-4
    dens_p0 = dens_pw + mm((b0 + mm((b1 + mm((b2 + mm((b3 + (b4*tempC)), tempC)), tempC)), tempC)), sal) + mm((c0 + mm((c1 + (c2*tempC)), tempC)), mm(sal, msqrt(sal))) + (d0*ms(sal))
    # Clear memory
    del b0, b1, b2, b3, b4, c0, c1, c2, d0, dens_pw
    #################################################
                                                                                                                                                        
    ### Compute the secant bulk modulus ###
    # Set constants and compute the compression terms
    h0 =  3.239908
    h1 =  1.43713e-3
    h2 =  1.16092e-4
    h3 = -5.77905e-7
    AW = h0 + mm((h1 + mm((h2 + (h3*tempC)), tempC)), tempC)
    # Clear memory
    del h0, h1, h2, h3

    k0 =  8.50935e-5
    k1 = -6.12293e-6
    k2 =  5.2787e-8
    BW = k0 + mm((k1 + (k2*tempC)), tempC)
    # Clear memory
    del k0, k1, k2

    e0 =  19652.21
    e1 =  148.4206
    e2 = -2.327105
    e3 =  1.360477e-2
    e4 = -5.155288e-5
    KW = e0 + mm((e1 + mm((e2 + mm((e3 + (e4*tempC)), tempC)), tempC)), tempC)
    # Clear memory
    del e0, e1, e2, e3, e4

    # Set constants and compute sea water terms
    i0 =  2.2838e-3    
    i1 = -1.0981e-5
    i2 = -1.6078e-6
    j0 =  1.91075e-4
    A = AW + mm((i0 + mm((i1 + (i2*tempC)), tempC)), sal) + (j0*mm(sal, msqrt(sal)))
    # Clear memory
    del i0, i1, i2, j0, AW

    m0 = -9.9348e-7
    m1 =  2.0816e-8
    m2 =  9.1697e-10
    B = BW + mm((m0 + mm((m1 + (m2*tempC)), tempC)), sal)
    # Clear memory
    del m0, m1, m2, BW

    f0 =  54.6746
    f1 = -0.603459
    f2 =  1.09987e-2
    f3 = -6.1670e-5
    g0 =  7.944e-2
    g1 =  1.6483e-2
    g2 = -5.3009e-4
    K0 = KW + mm((f0 + mm((f1 + mm((f2 + (f3*tempC)), tempC)), tempC)), sal) + mm((g0 + mm((g1 + (g2*tempC)), tempC)),  mm(sal, msqrt(sal))) 
    # Clear memory
    del f0, f1, f2, f3, g0, g1, g2, KW

    # Compute the secant bulk modulus
    K = K0 + mm((A + mm(B, pres)), pres)
    del A, B, K0
    #######################################

    # Compute the density
    dens = md(dens_p0, (1 - md(pres, K)))
    del dens_p0, K

    # Pass back the density
    return dens

# Function to generate a 3D array of alpha values.  Shp corresponds to the size of the 3D array and trans_mode sets the type of array to build
def voxelAlpha(shp, trans_mode):
    # Initialize the alpha 3D array
    alpha = np.zeros(shp)
    # For trans_mode == 0, build an array that gets more opaque towards the center
    if trans_mode == 0:
        # Create alpha vectors for the x, y and z directions
        # These vectors range from 0< to 1 and tend towards one at the center of the vector and tend to zero at the ends of the vector, linearly
        alpha_X = alphaVec(shp[0])
        alpha_Y = alphaVec(shp[1])
        alpha_Z = alphaVec(shp[2])

        # Loop over the i indicies (x vector/ x layer in the 3D array)
        for i in range(alpha.shape[0]):
            # Loop over the j indicies (y vector/ y layer in the 3D array)
            for j in range(alpha.shape[1]):
                # Loop over the k indicies (z vector/ z layer in the 3D array)
                for k in range(alpha.shape[2]):
                    # Set the 3D alpha array equal to the multiplication of each alpha vector in x y and z
                    alpha[i, j, k] = alpha_X[i]*alpha_Y[j]*alpha_Z[k]
    
    # For trans_mode == 1, build an array that gets more opaque towards the center along the jth index but is identical along the ith and kth index
    elif trans_mode == 1:
        # Create alpha vector for the lateral directions
        # This vector ranges from 0< to 1 and tend towards one at the center of the vector and tend to zero at the ends of the vector, linearly
        alpha_Y = alphaVec(shp[1])

        # Loop over the i indicies (thalwag/downstream vector in the 3D array)
        for i in range(alpha.shape[0]):
            # Loop over the k indicies (depth vector in the 3D array)
            for k in range(alpha.shape[2]):
                # At each depth in the depth vector and point along the thalwag vector, copy the alpha_Y vector
                alpha[i, :, k] = alpha_Y

    # Multiply the array by 0.9 to lower the alpha values (1 = solid and distorts the plotly figures)
    alpha = alpha*0.9
    # Return the 3D array
    return alpha

# Function to generate a linearly increasing to center/linearly decresing away from center alpha (alpha values range from 0 to 1) vector based on s, the length of the vector
def alphaVec(s):
    # Check if s is even...
    if s%2 == 0:
        # Set the step size based on s and the upper and lower limits for use with np.arange
        # If s is even then the step size will be 2*(1/s)
        step = 2/s
        # Make the upper limit a little larger
        ulim = 1 + (step/2)
        # Set the lower limit to the step size
        llim = step
        # Make the first half of the vector
        larr = np.arange(llim, ulim, step)
        # Make the second half of the vector (reverse of the first) and append it to the first vector
        ss = np.append(larr, np.flip(larr, 0))
    # If s is odd...
    else:
        # Set the step size based on s and the upper and lower limits for use with np.arange
        # If s is odd then the step size will be 2*(1/s+1)
        step = 2/(s + 1)
        # Make the upper limit a little larger
        ulim = 1 + (step/2)
        # Set the lower limit to the step size
        llim = step
        # Make the first half of the vector
        larr = np.arange(llim, ulim, step)
        # Make the second half of the vector (reverse of the first minus the first index) and append it to the first vector
        ss = np.append(larr, np.flip(larr, 0)[1:])

    # Return the alpha vector
    return ss

# Function to generate cuboid surfaces for 3D volumetric data (ax = axis artist, x/y/z = axis grid meshes, dat = grid-ed data, alpha = opacity grid, cmap = colormap for specific data type, cbnd = data bounds for each color in the colormap)
def voxelMat(ax, x, y, z, dat, alpha, cmap, cbnd):
    # Bin the data (dat) by indices corresponding to the boundaries given (cbnd)
    bin_idx = indexByBin(dat, cbnd)

    # Loop over the bin indice container
    for p, idx in enumerate(bin_idx):
        # Unpack the indice list
        idx = idx[0]

        # If the indice list for this bin is not empty ...
        if len(idx) != 0:
            # Build an array of cartesian coordinates in R3 using the indices for this bin
            f = lambda i: [x[i], y[i], z[i]]
            pos = np.asarray([f(i) for i in idx])

            # Hold on to the shapes of the mesh 
            x_end = x.shape[0] - 1
            y_end = x.shape[1] - 1
            z_end = x.shape[2] - 1

            # Initialize the corners array
            corners = np.zeros((pos.shape[0], 3, 8))
            # Initialize the alpha list
            alp = np.zeros(pos.shape[0])

            # Compute and store the x, y and z corners in a numpy array/vector
            # Using our point as the center of a cube, get staggered points from the neighboring cells
            # flc - front left corner
            # brc - back right corner
            # Start looping over the points
            for jj, dex in enumerate(idx):
                i = dex[0]
                j = dex[1]
                k = dex[2]
                # If the point is at the start of the thalwag...
                if i == 0:
                    # If the point is on the left bank edge...
                    if j == 0:
                        # Front points
                        flc_x, flc_y = centroid_2D([x[(i + 1), j, k], x[i, j, k]], [y[(i + 1), j, k], y[i, j, k]])
                        frc_x, frc_y = centroid_2D([x[(i + 1), (j + 1), k], x[(i + 1), j, k], x[i, (j + 1), k], x[i, j, k]], [y[(i + 1), (j + 1), k], y[(i + 1), j, k], y[i, (j + 1), k], y[i, j, k]])
                        # Back points
                        blc_x, blc_y = centroid_2D([x[i, j, k]], [y[i, j, k]])
                        brc_x, brc_y = centroid_2D([x[i, (j + 1), k], x[i, j, k]], [y[i, (j + 1), k], y[i, j, k]])
                    # If the point is on the right bank edge...
                    elif j == y_end:
                        # Front points
                        flc_x, flc_y = centroid_2D([x[(i + 1), (j - 1), k], x[(i + 1), j, k], x[i, (j - 1), k], x[i, j, k]], [y[(i + 1), (j - 1), k], y[(i + 1), j, k], y[i, (j - 1), k], y[i, j, k]])
                        frc_x, frc_y = centroid_2D([x[(i + 1), j, k], x[i, j, k]], [y[(i + 1), j, k], y[i, j, k]])
                        # Back points
                        blc_x, blc_y = centroid_2D([x[i, (j - 1), k], x[i, j, k]], [y[i, (j - 1), k], y[i, j, k]])
                        brc_x, brc_y = centroid_2D([x[i, j, k], ], [y[i, j, k]])
                    # If the point is inbetween the bank edges...
                    else:
                        # Front points
                        flc_x, flc_y = centroid_2D([x[(i + 1), (j - 1), k], x[(i + 1), j, k], x[i, (j - 1), k], x[i, j, k]], [y[(i + 1), (j - 1), k], y[(i + 1), j, k], y[i, (j - 1), k], y[i, j, k]])
                        frc_x, frc_y = centroid_2D([x[(i + 1), (j + 1), k], x[(i + 1), j, k], x[i, (j + 1), k], x[i, j, k]], [y[(i + 1), (j + 1), k], y[(i + 1), j, k], y[i, (j + 1), k], y[i, j, k]])
                        # Back points
                        blc_x, blc_y = centroid_2D([x[i, (j - 1), k], x[i, j, k]], [y[i, (j - 1), k], y[i, j, k]])
                        brc_x, brc_y = centroid_2D([x[i, (j + 1), k], x[i, j, k]], [y[i, (j + 1), k], y[i, j, k]])
                # If the point is at the end of the thalwag...
                elif i == x_end:
                    # If the point is on the left bank edge...
                    if j == 0:
                        # Front points
                        flc_x, flc_y = centroid_2D([x[i, j, k]], [y[i, j, k]])
                        frc_x, frc_y = centroid_2D([x[i, (j + 1), k], x[i, j, k]], [y[i, (j + 1), k], y[i, j, k]])
                        # Back points
                        blc_x, blc_y = centroid_2D([x[(i - 1), j, k], x[i, j, k]], [y[(i - 1), j, k], y[i, j, k]])
                        brc_x, brc_y = centroid_2D([x[(i - 1), (j + 1), k], x[(i - 1), j, k], x[i, (j + 1), k], x[i, j, k]], [y[(i - 1), (j + 1), k], y[(i - 1), j, k], y[i, (j + 1), k], y[i, j, k]])
                    # If the point is on the right bank edge...
                    elif j == y_end:
                        # Front points
                        flc_x, flc_y = centroid_2D([x[i, (j - 1), k], x[i, j, k]], [y[i, (j - 1), k], y[i, j, k]])
                        frc_x, frc_y = centroid_2D([x[i, j, k]], [y[i, j, k]])
                        # Back points
                        blc_x, blc_y = centroid_2D([x[(i - 1), (j - 1), k], x[(i - 1), j, k], x[i, (j - 1), k], x[i, j, k]], [y[(i - 1), (j - 1), k], y[(i - 1), j, k], y[i, (j - 1), k], y[i, j, k]])
                        brc_x, brc_y = centroid_2D([x[(i - 1), j, k], x[i, j, k]], [y[(i - 1), j, k], y[i, j, k]])
                    # If the point is inbetween the bank edges...
                    else:
                        # Front points
                        flc_x, flc_y = centroid_2D([x[i, (j - 1), k], x[i, j, k]], [y[i, (j - 1), k], y[i, j, k]])
                        frc_x, frc_y = centroid_2D([x[i, (j + 1), k], x[i, j, k]], [y[i, (j + 1), k], y[i, j, k]])
                        # Back points
                        blc_x, blc_y = centroid_2D([x[(i - 1), (j - 1), k], x[(i - 1), j, k], x[i, (j - 1), k], x[i, j, k]], [y[(i - 1), (j - 1), k], y[(i - 1), j, k], y[i, (j - 1), k], y[i, j, k]])
                        brc_x, brc_y = centroid_2D([x[(i - 1), (j + 1), k], x[(i - 1), j, k], x[i, (j + 1), k], x[i, j, k]], [y[(i - 1), (j + 1), k], y[(i - 1), j, k], y[i, (j + 1), k], y[i, j, k]])
                # The point is inbetween the ends of the thalwag...
                else:
                    # If the point is on the left bank edge...
                    if j == 0:
                        # Front points
                        flc_x, flc_y = centroid_2D([x[(i + 1), j, k], x[i, j, k]], [y[(i + 1), j, k], y[i, j, k]])
                        frc_x, frc_y = centroid_2D([x[(i + 1), (j + 1), k], x[(i + 1), j, k], x[i, (j + 1), k], x[i, j, k]], [y[(i + 1), (j + 1), k], y[(i + 1), j, k], y[i, (j + 1), k], y[i, j, k]])
                        # Back points
                        blc_x, blc_y = centroid_2D([x[(i - 1), j, k], x[i, j, k]], [y[(i - 1), j, k], y[i, j, k]])
                        brc_x, brc_y = centroid_2D([x[(i - 1), (j + 1), k], x[(i - 1), j, k], x[i, (j + 1), k], x[i, j, k]], [y[(i - 1), (j + 1), k], y[(i - 1), j, k], y[i, (j + 1), k], y[i, j, k]])
                    # If the point is on the right bank edge...
                    elif j == y_end:
                        # Front points
                        flc_x, flc_y = centroid_2D([x[(i + 1), (j - 1), k], x[(i + 1), j, k], x[i, (j - 1), k], x[i, j, k]], [y[(i + 1), (j - 1), k], y[(i + 1), j, k], y[i, (j - 1), k], y[i, j, k]])
                        frc_x, frc_y = centroid_2D([x[(i + 1), j, k], x[i, j, k]], [y[(i + 1), j, k], y[i, j, k]])
                        # Back points
                        blc_x, blc_y = centroid_2D([x[(i - 1), (j - 1), k], x[(i - 1), j, k], x[i, (j - 1), k], x[i, j, k]], [y[(i - 1), (j - 1), k], y[(i - 1), j, k], y[i, (j - 1), k], y[i, j, k]])
                        brc_x, brc_y = centroid_2D([x[(i - 1), j, k], x[i, j, k]], [y[(i - 1), j, k], y[i, j, k]])
                    # If the point is inbetween the bank edges...
                    else:
                        # Front points
                        flc_x, flc_y = centroid_2D([x[(i + 1), (j - 1), k], x[(i + 1), j, k], x[i, (j - 1), k], x[i, j, k]], [y[(i + 1), (j - 1), k], y[(i + 1), j, k], y[i, (j - 1), k], y[i, j, k]])
                        frc_x, frc_y = centroid_2D([x[(i + 1), (j + 1), k], x[(i + 1), j, k], x[i, (j + 1), k], x[i, j, k]], [y[(i + 1), (j + 1), k], y[(i + 1), j, k], y[i, (j + 1), k], y[i, j, k]])
                        # Back points
                        blc_x, blc_y = centroid_2D([x[(i - 1), (j - 1), k], x[(i - 1), j, k], x[i, (j - 1), k], x[i, j, k]], [y[(i - 1), (j - 1), k], y[(i - 1), j, k], y[i, (j - 1), k], y[i, j, k]])
                        brc_x, brc_y = centroid_2D([x[(i - 1), (j + 1), k], x[(i - 1), j, k], x[i, (j + 1), k], x[i, j, k]], [y[(i - 1), (j + 1), k], y[(i - 1), j, k], y[i, (j + 1), k], y[i, j, k]])

                # If the point is at the water surface...
                if k == 0:
                    top_z = z[i, j, k]
                    bottom_z = (z[i, j, (k + 1)] + z[i, j, k])/2 
                # If the point is at the deepest depth...
                elif k == z_end:
                    top_z = (z[i, j, (k - 1)] + z[i, j, k])/2 
                    bottom_z = z[i, j, k]
                # If the point is inbetween the water surface and deepest depth...
                else:
                    top_z = (z[i, j, (k - 1)] + z[i, j, k])/2 
                    bottom_z = (z[i, j, (k + 1)] + z[i, j, k])/2

                # Store the corners in the following order (for use with the surface definition function voxelSurf)
                # X coordinates [0, 0, 1, 1, 0, 0, 1, 1]
                # Y coordinates [0, 1, 1, 0, 0, 1, 1, 0]
                # Z coordinates [0, 0, 0, 0, 1, 1, 1, 1]
                # x = [blx, brx, frx, flx, blx, brx, frx, flx], 
                # y = [bly, bry, fry, fly, bly, bry, fry, fly], 
                # z = [tz, tz, tz, tz, bz, bz, bz, bz]
                corners[jj, :, :] = [blc_x, brc_x, frc_x, flc_x, blc_x, brc_x, frc_x, flc_x], [blc_y, brc_y, frc_y, flc_y, blc_y, brc_y, frc_y, flc_y], [top_z, top_z, top_z, top_z, bottom_z, bottom_z, bottom_z, bottom_z]
                # Store the alpha value for this index
                alp[jj] = alpha[i, j, k]

            # Pass the corners, colormap, alpha array and axis artist to the voxel function to generate the cube surfaces
            #   and plot it 
            ax = voxel(corners, cmap[p], alp, ax)
            p = p + 1
    
    # Return the artist
    return ax

# Function to invoke the generation of a cubes surfaces from its verticies given one surface color and varying transparency based on position
#   and plot it to an axis artist (corners = spatial verticies for each cube, c = color, a = opacity for each cube, ax = axis artist)
def voxel(corners, c, a, ax):
    # Inact the surface face definitions (PLOTLY)
    surf_pltly = voxelSurf()

    # Get unique values of the opacity array and loop over them
    uniq = np.unique(a)
    for u in uniq:
        # (Re-)Initialized the grouped corner arrays xCombined, yCombined and zCombined
        xCombined = []
        yCombined = []
        zCombined = []
        # (Re-)Initialize the grouped face indexing arrays
        iCombined = []
        jCombined = []
        kCombined = []

        # Store the indices where the unique opacity value lives in a
        idx = np.where(a == u)[0]
        # Loop over the indice set
        for cnt, i in enumerate(idx):
            xCombined.extend(corners[i][0].tolist())
            yCombined.extend(corners[i][1].tolist())
            zCombined.extend(corners[i][2].tolist())

            cntMod = cnt*8
            iMod = surf_pltly[0] + cntMod
            iCombined.extend(iMod.tolist())
            jMod = surf_pltly[1] + cntMod
            jCombined.extend(jMod.tolist())
            kMod = surf_pltly[2] + cntMod
            kCombined.extend(kMod.tolist())

        # Make the cubes as a Mesh3d object
        cubes = go.Mesh3d(
                        x = xCombined,
                        y = yCombined,
                        z = zCombined,
                        opacity = u,
                        color = '#%02x%02x%02x' % (int(c[0]*255), int(c[1]*255), int(c[2]*255)),
                        i = iCombined,
                        j = jCombined,
                        k = kCombined,
                        )
        # Append the object to the axis artist
        ax.append(cubes)

    # Pass back the axis artist
    return ax

# Function to define surfaces for a cube in R3 by referencing the 
#   indicies for the vertex sets stored in a list; face definitions (FOR PLOTLY)
def voxelSurf():
    # Initialize an array the size of 3 with a list of 12(triangular faces per cube)
    f = np.zeros((3, 12))

    # The i, j, k face arrays below work as follows:
    #   If [i[0], j[0], k[0]] = [7, 3, 0] then 
    #   the 0th face has the verticies [1, 0, 1], [1, 0, 0], [0, 0, 0]
    #                                    (7th)      (3rd)      (0th)
    #                    [x[7], y[7], z[7]], [x[3], y[3], z[3]], [x[0], y[0], z[0]]
    # i-th face
    f[0, :] = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
    # j-th face
    f[1, :] = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
    # k-th face
    f[2, :] = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]
    
    # Return the packaged face definitions
    return f

# Function to determine the euclidean distance between each point in 2D space (x being a vector in one dimension, y being a vector in the second dimension)
def euc_dist(x, y, z = np.asarray([])):
    # Check if data is in R2
    if (z.size == 0):
        # Verify x and y are of the same shape
        if x.shape == y.shape:
            # Pair the spline coordinates
            loc = np.array((x, y)).T
            # Initialize the difference vectors
            diffx = np.zeros(len(loc))
            diffy = np.zeros(len(loc))
            # Compute the distance between each point
            for i in range(1, len(loc), 1):
                # Calculate the difference between x1,x0 and y1,y0
                diffx[i], diffy[i] = loc[i] - loc[i - 1]
            # Return the distance array
            return msqrt(ms(diffx) + ms(diffy))
        # X and y were not of the same shape
        else:
            print('Failed at line %i.' % inspect.currentframe().f_back.f_lineno)
            print("The input x and y must be of equal length.")
            # Stop execution
            sys.exit(0)
    # Check data is in R3
    else:
        if x.shape == y.shape == z.shape:
            loc = np.array((x, y, z)).T
            diffx = np.zeros(len(loc))
            diffy = np.zeros(len(loc))
            diffz = np.zeros(len(loc))
            for i in range(1, len(loc), 1):
                diffx[i], diffy[i], diffz[i] = loc[i] - loc[i - 1]
            return msqrt(ms(diffx) + ms(diffy) + ms(diffz))
        # X, Y and Z were not of the same shape
        else:
            print('Failed at line %i.' % inspect.currentframe().f_back.f_lineno)
            print("The input x, y and z must be of equal length.")
            # Stop execution
            sys.exit(0)

# Function to break a splined curve (x, y) into equally spaced, by distance (dd), points on a curve with a total length of td
#   Added modification to output normal slope-intercept line coefficients (m and b) at each equally spaced point
def equal_dist(x, y, td, dd):
    # Get number of points given the number of line segments plus one
    pnts = int((td/dd) + 1)

    # Initialize the output point sets
    x_out = np.zeros(pnts)
    y_out = np.zeros(pnts)
    m_norm = np.zeros(pnts)
    b_norm = np.zeros(pnts)

    # Group the input point set
    loc = np.array((x, y)).T
    # Initialize a counter variable
    k = 0

    # Loop over the input point set
    for i, j in enumerate(loc):
        # If not the first point the input set...
        if (i >= 1) and (k < pnts): 
            # Find distance from the seg_start
            diff = loc[i] - seg_start
            dist = (diff[0]**2 + diff[1]**2)**.5
            # If the distance is greater than the dd spacing...
            if dist > dd:
                # Get the distance from the seg_start for the previous point
                diff2 = loc[i-1] - seg_start
                dist2 = (diff2[0]**2 + diff2[1]**2)**.5
                # Find the distance from the previous point to the new point that is dd far from the seg_start
                dist3 = dd - dist2
                # Find the distance from the current point to the new point that is dd far from the seg_start
                dist4 = dist - dd

                # Visualization
                # start-------------xy1--------xyn--------xy2
                # |-----dist------------------------------|
                # |-----dist2-------|
                # |-----dd---------------------|
                #                   |---dist3--|
                #                              |---dist4--|

                # Solve for the point xn, yn and get the normal slope-int coeff
                xn, yn, mn, bn = arb_pnt_btwn(dist3, loc[i-1][0], loc[i-1][1], dist4, loc[i][0], loc[i][1])
                # Update the segment start 
                seg_start = np.array([xn, yn])
                # Assign the even spaced points to the output arrays
                x_out[k] = xn
                y_out[k] = yn
                # Set the normal slope-intercept line coefficients
                m_norm[k] = mn
                b_norm[k] = bn

                # Next counter/index
                k = k + 1

        # If the number of output points has been reached, exit the loop
        elif k >= pnts:
            break
        # For the first point in the input set...
        else:
            # Set the first point location as seg_start
            seg_start = loc[i]
            # Assign the first point to the x and y ouput arrays
            x_out[k] = loc[i][0]
            y_out[k] = loc[i][1]
            # Get the normal line-intercept coefficients
            m_norm[k] = -1*((loc[i + 1][0] - loc[i][0])/(loc[i + 1][1] - loc[i][1]))
            b_norm[k] = loc[i][1] - (m_norm[k]*loc[i][0])
            # Add to the output counter/index
            k = k + 1
    
    # Return the evenly spaced set of points along the splined thalwag
    return x_out, y_out, m_norm, b_norm

# Function to find the nearest number to m that divides n with no remainder
def find_div(n, m, i):
    # Round m to the nearest value divisible by i
    m = round(m/i)*i
    # Get upper nearest number to m that divides n
    m1 = fdu(n, m, i)
    # Get lower nearest number to m that divides n
    m2 = fdd(n, m, i)

    # Find if m1 is closer to m than m2...
    if abs(m - m1) <= abs(m - m2):
        # If so, return m1
        return m1
    # Find if m2 is closer to m than m2...
    elif abs(m - m2) < abs(m - m1):
        # If so, return m2
        return m2

# Recursive function to find nearest number above m that divides n with no remainder
def fdu(n, m, i):
    # if modulus doesnt equal 0
    if (n % m) != 0:
        # call function again using m + i
        return fdu(n, (m + i), i)
    # if modulus equals 0
    elif (n % m) == 0:
        # return new m
        return m

# Recursive function to find the nearest number below m that divides n with no remainder
def fdd(n, m, i):
    # if modulus doesnt equal 0
    if (n % m) != 0:
        # call function again using m + i
        return fdd(n, (m - i), i)
    # if modulus equals 0
    elif (n % m) == 0:
        # return new m
        return m

# Function to find a parametric point that fits on the same line as (x1,y1) and (x2,y2) AND is d1 from (x1,y1) AND is d2 from (x2,y2)
#   Returns the point that meets this criteria (xn, yn) and the line-intercept coeff. for a perpindicular line to the line through (x1,y1) and (x2,y2) while intersecting at (xn,yn)
def arb_pnt_btwn(d1, x1, y1, d2, x2, y2):
    # Convergence control 
    con = 0.5

    # Put a line through the points 1 and 2
    m = (y2 - y1)/(x2 - x1)
    b = y1 - (m*x1)
    
    # Using the equations d = sqrt((xn - x1)^2 + (yn - y1)^2) and yn = (m*xn) + b solve for xn by plugging in for yn into the first equation
    # Set coefficients for the quadratic formula
    A = (1 + (m**2))
    B = (2*b*m) - (2*x1) - (2*m*y1)
    C = (d1**2) + (2*b*y1) - (x1**2) - (b**2) - (y1**2) 
    
    # If A is not equal to 0
    if A != 0:
        # Get the two solutions for the quadratic
        xi = ((-1*(((4*A*C) + (B**2))**.5)) - B)/(2*A)
        xii = ((((4*A*C) + (B**2))**.5) - B)/(2*A)

        # Solve for y by plugging xn into the second equation mentioned before
        yi = (m*xi) + b
        yii = (m*xii) + b

        # Solve for the distances between x1 and x2 using xn and yn now for both solutions
        d1i = ((((xi - x1)**2) + ((yi - y1)**2))**0.5)
        d2i = ((((xi - x2)**2) + ((yi - y2)**2))**0.5)
        d1ii = ((((xii - x1)**2) + ((yii - y1)**2))**0.5)
        d2ii = ((((xii - x2)**2) + ((yii - y2)**2))**0.5)

        # Verify if solution one is the correct solution
        if (abs(d1i - d1) < con) and (abs(d2i - d2) < con):
            # Assign the even spaced points to the output arrays
            xn = xi
            yn = yi
        # Verify if solution two is the correct solution
        elif (abs(d1ii - d1) < con) and (abs(d2ii - d2) < con):
            # Assign the even spaced points to the output arrays
            xn = xii
            yn = yii
        # Solution doesnt exist/Error
        else:
            # Print issue
            print('Failed at line %i.' % inspect.currentframe().f_back.f_lineno)
            print("Function arb_pnt_btwn crashed: Solution doesnt exist/Solution Error.  Values may be to large to compute.")
            # Stop execution
            sys.exit(0)

    # If A is equal to 0
    elif A == 0:
        # If B is not equal to 0 (if B did equal 0 there would be an infinite number of solutions)
        if B != 0:
            # Using the equation 0 = Axn**2 + Bxn - C when A = 0 yields 0 = Bxn - C ---> xn = C/B 
            xi = -1*(C/B)

            # Solve for y by plugging xn into the second equation mentioned before
            yi = (m*xi) + b

            # Solve for the distances between x1 and x2 using xn and yn now 
            d1i = ((((xi - x1)**2) + ((yi - y1)**2))**.5)
            d2i = ((((xi - x2)**2) + ((yi - y2)**2))**.5)
            
            # Verify that the solution is correct
            if (abs(d1i - d1) < con) and (abs(d2i - d2) < con):
                # Assign the even spaced points to the output arrays
                xn = xi
                yn = yi
            # The solution is an infinite set of points
            else:
                # Print issue
                print('Failed at line %i.' % inspect.currentframe().f_back.f_lineno)
                print("Function arb_pnt_btwn crashed: Solution is an infinite comibination of points. Error.")
                # Stop execution
                sys.exit(0)
                
    # Get the normal slope-intercept line coefficients
    mn = (-1/m)
    bn = yn - (xn*mn)
    
    # Return xn, yn, mn and bn
    return xn, yn, mn, bn

# Function to find a parametric point that fits on the line y = mx + b that intersects (x1, y1) and is d from (x1, y1).  x2 helps to find if the solution points are on the left or right side of a downstream pointing vector.
#   Returns the points that meet this criteria (xi, yi) and (xii, yii) 
#   Output is ordered where the first x,y pair is on the left and the second x,y pair is on the right
def arb_pnt_awy(d1, x1, y1, m, b, x2, y2):  
    # Convergence control 
    con = 0.5

    # Using the equations d = sqrt((xn - x1)^2 + (yn - y1)^2) and yn = (m*xn) + b solve for xn by plugging in for yn into the first equation
    # Set coefficients for the quadratic formula
    A = (1 + (m**2))
    B = (2*b*m) - (2*x1) - (2*m*y1)
    C = (d1**2) + (2*b*y1) - (x1**2) - (b**2) - (y1**2) 
    
    # If A is not equal to 0
    if A != 0:
        # Get the two solutions for the quadratic
        xi = ((-1*(((4*A*C) + (B**2))**.5)) - B)/(2*A)
        xii = ((((4*A*C) + (B**2))**.5) - B)/(2*A)

        # Solve for y by plugging xn into the second equation mentioned before
        yi = (m*xi) + b
        yii = (m*xii) + b

        # Solve for the distances between x1 and x2 using xn and yn now for both solutions
        d1i = ((((xi - x1)**2) + ((yi - y1)**2))**0.5)
        d1ii = ((((xii - x1)**2) + ((yii - y1)**2))**0.5)

        # Verify if solutions are correct
        if (abs(d1i - d1) < con) and (abs(d1ii - d1) < con):
            # Assign the even spaced points to the output arrays
            xs1 = xi
            ys1 = yi
            xs2 = xii
            ys2 = yii
        # Solution doesnt exist/Error
        else:
            # Print issue
            print('Failed at line %i.' % inspect.currentframe().f_back.f_lineno)
            print("Function arb_pnt_awy crashed: Solution doesnt exist/Solution Error.  Values may be to large to compute.")
            # Stop execution
            sys.exit(0)

    # If A is equal to 0
    elif A == 0:
        # Print issue
        print('Failed at line %i.' % inspect.currentframe().f_back.f_lineno)
        print("Function arb_pnt_awy crashed: Coefficient A equals zero indicating that there is one solution.  This cannot be true.  There are always two points on a line at a distance d from a specific point on the same line.  Error.")
        # Stop execution
        sys.exit(0)
                  
    # Determine if the points are on the left or right 
    # Use the determinant of the vectors p1->p2 and p1->ps1
    pos = np.sign(((x2 - x1)*(ys1 - y1)) - ((y2 - y1)*(xs1 - x1)))

    # If pos = -1 then the point s1 is on the right...
    if pos < 0:
        # Return the points in order of left point and right point
        return xs2, ys2, xs1, ys1 
    # Else if pos = +1 then the point s1 is on the left...
    elif pos > 0:
        # Return the points in order of left point and right point
        return xs1, ys1, xs2, ys2

# Function to build a meshgrid for the curvilinear system .  [tx, ty] are the coordinates for the points along the thalwag, indexing with xg (the thalwag distance axis).
#   [m, b] are the slope-intercept coefficients for a line that is perpindicular along the thalwag track.
#   xg is the thalwag axis vector (upstream to downstream), yg is the lateral axis vector (left bank to right bank going downstream) and zg is the vertical axis vector (water surface to deepest point)
def curvey_grid(tx, ty, m, b, xg, yg, zg):
    # Initialize the grid mesh arrays
    xg_msh = np.zeros((xg.shape[0], yg.shape[0], zg.shape[0]))
    yg_msh = np.zeros((xg.shape[0], yg.shape[0], zg.shape[0]))
    zg_msh = np.zeros((xg.shape[0], yg.shape[0], zg.shape[0]))

    # Second loop control variable
    half_j = ((yg.shape[0] - 1)/2)
    # Since the vertical axis is independent of position we can build one vertical grid and assign it to the different indicies
    zg_tmp = np.tile(zg, (yg.shape[0], 1))

    # Loop over the points along the thalwag axis
    for i, (px, py, mm, bb) in enumerate(zip(tx, ty, m, b)):
        ### This section is for indexing/determining the side of the thalwag line that the lateral points are on 
        # If the loop is at the first thalwag point in tx, ty...
        if i == 0:
            # Check if y is increasing or decreasing
            ytrnd = np.sign(ty[i + 1] - ty[i])
            # Get a line slope-intercept function for the last and next to last thalwag points
            trnd = (ty[i + 1] - ty[i])/(tx[i + 1] - tx[i])
            intr = ty[i] - (trnd*tx[i])
        # If the loop is not at the first thalwag point in tx, ty...
        else:
            # Check if y is increasing or decreasing
            ytrnd = np.sign(ty[i] - ty[i - 1])
            # Get a line slope-intercept function for the last and next to last thalwag points
            trnd = (ty[i] - ty[i - 1])/(tx[i] - tx[i - 1])
            intr = ty[i] - (trnd*tx[i])

        # If y is increasing ...
        if ytrnd > 0:
            # Use the slope-intercept function to forward predict a x coordinate downstream
            # Alter y based on how the previous points were trending (+ 10)
            py_next = ty[i] + 10
        # If y is decreasing ...
        elif ytrnd < 0:
            # Use the slope-intercept function to forward predict a x coordinate downstream
            # Alter y based on how the previous points were trending (- 10)
            py_next = ty[i] - 10
        # Use the function to get an x coordinate
        px_next = ((py_next) - intr)/trnd
        ### We will feed px_next and py_next into the arb_pnt_awy function later on

        # Loop over the lateral positions on a perpindicular line that intersects the thalwag point 
        for j, y in enumerate(yg):
            # If the jth index below its half point...
            if j < half_j:
                # Find the x and y positions on the mesh that are abs(y) far from the thalwag coordinates along the line y = mm*x + bb
                #   One x,y pair will be on the left side and the other pair on the right (two solutions)
                xg_msh[i, j, :], yg_msh[i, j, :], xg_msh[i, (-1 - j), :], yg_msh[i, (-1 - j), :] = arb_pnt_awy(abs(y), px, py, mm, bb, px_next, py_next)
            # If the jth index is at the half point...
            elif j == half_j:
                # Assign the central point of the mesh as the thalwag point
                xg_msh[i, j, :], yg_msh[i, j, :] = px, py
                # Break out of the loop
                break

        # Assign the vertical mesh to this index on the thalwag axis
        zg_msh[i, :, :] = zg_tmp

    # Return the curvilinear grid mesh
    return xg_msh, yg_msh, zg_msh

# Function to get the centroid of a set of points on a 2D plane (i being one axis, j being the other)
def centroid_2D(i, j):
    # Go over the i and j axis point sets and sum them
    sum_i = np.sum(i)
    sum_j = np.sum(j)

    # Divide the sums by the number of elements in each point set
    ci = sum_i/len(i)
    cj = sum_j/len(j)

    # Return the centroid point on the ij plane
    return ci, cj

# Function to make the required lists for plotly and matplotlib colorbars
def generateColorbar(map, bounds):
    # Generate a linear vector ranging from 0 to 1 with the length of the colormap boundaries
    cscale_num = np.linspace(0, 1, len(bounds))
    # Generate the rgb string vector from the colormap
    cscale_rgb = ['rgb(%3.f, %3.f, %3.f)' % ((rgb_map[0]*255), (rgb_map[1]*255), (rgb_map[2]*255)) for rgb_map in map]
    # Make the colorscale array for plotly
    scale_plotly = [[n, r] for n, r in zip(cscale_num, cscale_rgb)]

    # Make the colorscale array for matplotlib
    scale_matplotlib = map.tolist()

    # Remove the zero from the bounds
    cbar_tickval = bounds[1:]

    # Make the tick text for matplotlib with less than and greater than symbols in front of the 
    #   lowest and highest bounding value
    cbar_ticktext = tickval2text(cbar_tickval)

    # Return the matplotlib friendly colorscale array, bounds and the colorbar tick text labels
    return scale_matplotlib, cbar_tickval, cbar_ticktext, scale_plotly

# Function to turn tick values into labels, with < and > symbols at the tails
def tickval2text(val):
    # Turn all the tick values into strings
    txt = ['%.2f' % b for b in val]
    # Add a less than to the first tick string
    txt[0] = ' < ' + txt[0]
    # Add some padding to the inner tick strings
    txt[1:-1] = [' ' + b for b in txt[1:-1]] 
    # Add a greater than to the last tick string
    txt[-1] = ' > ' + txt[-1]

    # Return the stringafide tick labels
    return txt

# Function to make the required vectors for plotly streamtubes (feeding in tube start and end points)
#   The utility seems broken in regards to how the user assigns the start and end (bounds) of the 
#   streamtubes.  Directionality though seems correct. 
#   (xs, ys) denote the starting point of the arrow, (xe, ye) denote the ending point of the arrow
def arrow_stem(xs, ys, xe, ye):
    # Get the directional vectors
    umag = abs(xe - xs)
    vmag = abs(ye - ys)
    # Scale the directions
    scl = max(umag, vmag)
    uscl = (xe - xs)/scl
    vscl = (ye - ys)/scl

    # Make the x grid boundary
    # Bound must "touch" the starting x value and the other bound must be half the x distance between the x points
    #   Bounds must be sorted from lowest to highest 
    gx = xs + ((xe - xs)/2)
    if gx < xs:
        gxl = gx
        gxu = xs
    else:
        gxl = xs
        gxu = gx

    # Make the y grid boundary
    # Bound must "touch" the starting y value and the other bound must be half the y distance between the y points
    #   Bounds must be sorted from lowest to highest 
    gy = ys + ((ye - ys)/2)
    if gy < ys:
        gyl = gy
        gyu = ys
    else:
        gyl = ys
        gyu = gy

    # Make the mesh grid using the x an y bounds and set z's bounds from 0 to 1 (though the tube will be on the z = 0 plane)
    x, y, z = np.mgrid[gxl:gxu:2j, gyl:gyu:2j, 0:1:2j]
    # Make the directional grids for u and v using the u scale and v scale computed earlier
    u = np.ones(x.shape)*uscl
    v = np.ones(x.shape)*vscl
    # The w directional grid will be 0 everywhere
    w = np.zeros(x.shape)
    
    # Flatten all the mesh grids into vectors
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    u = u.flatten()
    v = v.flatten()
    w = w.flatten()

    # Return the x, y, z, u, v and w vectors for plotly's streamtubes
    return x, y, z, u, v, w

# Function to build a plotly layout object given axis titles, figure margins, axis ranges and axis tick values and labels
#   Additionally this figure will have a font of size 10 and family Universal Condensed
#   And a plot area bounded by a black 1 pt line
#   And tick marks will be inside the plot area
def lay_2d(f_title, x_title, y_title, f_margin, x_range, x_tickvals, x_ticktext, y_range, y_tickvals, y_ticktext, hover_format):
    # Set lay equal to a plotly layout graphics object with title
    lay = go.Layout(title = f_title,
        # Set the font size and family
        font = dict(
            family = 'UniversCond',
            size = 10
        ),
        # X-axis settings
        xaxis = dict(
            # Title
            title = x_title,
            # Range
            range = x_range,
            # Tick oreintation
            ticks = 'inside',
            # Values to place ticks
            tickvals = x_tickvals,
            # Text for ticks at those values
            ticktext = x_ticktext,
            # Axis line color
            linecolor = 'rgb(0, 0, 0)',
            # Axis line width
            linewidth = 1,
            # Axis line also shown on opposite side of plot area
            mirror = True
        ),
        # Y-axis settings
        yaxis = dict(
            # Title
            title = y_title,
            # Range
            range = y_range,
            # Tick oreintation
            ticks = 'inside',
            # Values to place ticks
            tickvals = y_tickvals,
            # Text for ticks at those values
            ticktext = y_ticktext,
            # Axis line color
            linecolor = 'rgb(0, 0, 0)',
            # Axis line width
            linewidth = 1,
            # Axis line also shown on opposite side of plot area            
            mirror = True,
            # Set the hover format
            hoverformat = hover_format
        ),
        # Figure margins
        margin = dict(
            # Left margin
            l = f_margin,
            # Right margin
            r = f_margin,
            # Top margin
            t = f_margin,
            # Bottom margin
            b = f_margin
        )
    )

    # Return the plotly layout object
    return lay 

# Function to perform a robust lowess filter on the input data
#   y, assumed to be sequential, using a window of size n away
#   from the central point
def robustLowess(y, f, iterations = 3):
    # Get the number of data points
    n = len(y)
    # Build an index list
    x = np.arange(n)
    # Set the total number of points in the window
    winWidth = int((f*2) + 1)
    # Define the index distance lists for each index move
    dist = np.abs(x[:, None] - x[None, :])
    # Define the largest window distance away for each index move
    maxDist = [np.sort(d)[winWidth] for d in dist]
    # Form the indice range lists for each index move
    jRange = [[int(lb), int(ub)] for lb, ub in zip(x - f, x + f + 1)]
    for j in range(n):
        if jRange[j][0] < 0:
            jRange[j][0] = 0
            jRange[j][1] = winWidth
        else:
            break
    for j in reversed(range(n)):
        if jRange[j][1] > n:
            jRange[j][0] = n - winWidth
            jRange[j][1] = n
        else:
            break
    # Initialize the lowess filtered data as nan
    yLowess = np.full(y.shape, np.nan)
    # Begin the iterations loop
    i = 0
    while i < iterations:
        # Step over each index
        for j, rng, d, h in zip(range(n), jRange, dist, maxDist):
            # Form the tricubic weight
            wt = d/h
            wt = np.clip(wt, 0.0, 1.0)
            wt = (1 - wt**3)**3
            # Subset the weight
            wt = wt[rng[0]:rng[1]]
            # If this is not first iteration set the robust weights 
            if i != 0:
                # Subset the residual
                r = residual[rng[0]:rng[1]]
                # Get the MAD value
                MAD = 6*np.sort(r, axis = None, kind = 'quicksort')[int(f)]
                # Create the robust weight coefficient
                G = r/MAD
                # Form a logic vector
                boo = np.zeros(winWidth)
                # Reform the robust weight
                boo[G < 1] = 1
                G = boo*((1 - G**2)**2)
                # Create the final weighting function
                wt = G*wt
            # Subset x and y
            X = x[rng[0]:rng[1]]
            Y = y[rng[0]:rng[1]]
            # Make a matrix of the weight vectors to be summed
            #                 [wt   wtx    wtx2     wty    wtxy  ]
            sMat = np.asarray([wt,  wt*X,  wt*X*X,  wt*Y,  wt*X*Y])
            # Sum along the rows
            sMat = np.nansum(sMat, axis = 1)
            # Solve for the least squares fit coefficients
            det = (sMat[0]*sMat[2]) - (sMat[1]*sMat[1])
            beta0 = ((sMat[2]*sMat[3]) - (sMat[1]*sMat[4]))/det
            beta1 = ((sMat[0]*sMat[4]) - (sMat[1]*sMat[3]))/det
            # Compute current lowess value
            yLowess[j] = beta0 + (x[j]*beta1) 
        # Compute the residual between the raw and smoothed data
        residual = abs(y - yLowess)
        # Next iteration
        i += 1

    # Return the smoothed data
    return yLowess

# Function to calculate the area of a triangle with
#   arbitrary coordinates
def areaArbTri(ax, ay, bx, by, cx, cy):
    area = abs(((ax*(by - cy)) + (bx*(cy - ay)) + (cx*(ay - by)))/2)
    return area

# Function to convert a matplotlib figure to a base64 encoded data URI
#   for use with the Dash application as an html image tag
def mplToURI(fig, close_all = True, **save_args):
    # Open a binary stream (memory buffer/chunk of RAM)
    img = BytesIO()
    # Save the matplotlib figure to the image stream
    fig.savefig(img, format = 'png', **save_args)
    if close_all:
        fig.clf()
        plt.close('all')
    # Go to the start of the image
    img.seek(0)
    # Apply a base64 encoding to the image stream
    encoded = base64.b64encode(img.read()).decode("ascii").replace("\n", "")

    # Return the data URI
    return "data:image/png;base64,{}".format(encoded)

# Function to generate a nested list of indices corresponding to where values in data
#   fall between the brackets established by bounds
def indexByBin(data, bounds):
    # Set up the bin endpoints using bounds
    bin_start = bounds[:]
    bin_end = bounds[1:]
    # Make the endpoints len+1 for data higher than the last bound
    bin_start.extend([bounds[-1]])
    bin_end.extend([bounds[-1]])

    # Make sure data is a numpy array
    data = np.asarray(data)
    # Check the shape of the data set
    if len(data.shape) == 1:
        # Make a list for all possible indices for data
        index_temp = np.arange(0, len(data), 1).tolist()
    elif len(data.shape) == 2:
        # Make a list of tuples for all possible indices for data
        index_temp = []
        f = lambda x: index_temp.append(x)
        [f((i, j)) for i in range(0, data.shape[0]) for j in range(0, data.shape[1])]
    elif len(data.shape) == 3:
        # Make a list of tuples for all possible indices for data
        index_temp = []
        f = lambda x: index_temp.append(x)
        [f((i, j, k)) for i in range(0, data.shape[0]) for j in range(0, data.shape[1]) for k in range(data.shape[2])]

    # Initialize an indice list
    index = []
    # Loop over the indices and remove nan values
    for ii in index_temp:
        if not np.isnan(data[ii]):
            index.append(ii)

    # Find negative data and set to 0
    for ii in index:
        if data[ii] < 0:
            data[ii] = 0

    # Initialize the container for the indices corresponding to data in each bin
    bin_idx = [[] for _ in range(len(bounds))]
    # Initialize a counter
    p = 0
    # Loop over bins
    for s, e in zip(bin_start, bin_end):
        # Initialize an indice list
        store_index = []
        # Check if there are remaining data points that have not already been assigned to a bin
        if len(index) != 0:
            # Copy the indice list and re-initialize it
            index_temp = index
            index = []
            # If we are at the last boundary ...
            if p == (len(bounds) - 1):
                # Remaining data values begotten by index_temp must be larger than e (the last bin)
                store_index = index_temp
            # If not ...      
            else:
                # Go over the current data points remaining ...
                for ii in index_temp:
                    # If the data is within the bounds store it ...
                    if ((data[ii] >= s) & (data[ii] < e)):
                        store_index.append(ii)
                    # If not then store the index back into the remaining indice list
                    else:
                        index.append(ii)

        # Store the indice list by its bin
        bin_idx[p].append(store_index)
        # Add to the counter
        p = p + 1
    
    # Return the nested lists of indices for the binned data
    return bin_idx

def randomColorHex():
    hexSymbols = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    hexColor = '#'
    for _ in range(6):
        hexColor = ''.join([hexColor, randomSample(hexSymbols, 1)[0]])
    return hexColor
########################################################################