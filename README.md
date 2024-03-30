# obs-extrapolation
Saccade is a weird OBS Python script that is weird

`Saccade attempts to make video recordings of perspective / widescreen games display more salient information when packed into portrait-form content. 
It can do this in real-time for streaming.`

It is primarily based on https://github.com/upgradeQ/OBS-Studio-Python-Scripting-Cheatsheet-obspython-Examples-of-API/blob/master/src/get_source_frame_data_ffi.py

It sort of works, but it applies smoothed proportional control correlated with scene motion and some other hacky control stuff.

It badly needs to be refactored and it also crashes unpredictably under various circumstances that I can't explain.

It depends on Python, OBS, Numpy, OpenCV

# Install Instructions
⚠ This software is free-of-charge without any warranty.

## Install Python
Grab Python portable and install it in your root directory

## Install Deps
Use a pip bootstrap to grab:
`cd C:\PythonPortable`
`$gp = wget https://bootstrap.pypa.io/get-pip.py`
`echo $gp.Content > get-pip.py`
`notepad.exe get-pip.py` (Save as ANSI)
`.\python.exe get-pip.py`
`.\python.exe -m pip install opencv-python`
`.\python.exe -m pip install opencv-contrib-python`

## Install OBS
Install OBS
Inside of Tools > Scripts go to Python tab
Point to C:\PythonPortable

## Make a scene
Default name for the Source is "Spaceship"
Create a source named "Spaceship" It probably has to be less than 1920x1080, for **reasons**
Make your Canvas portrait
Center and vertically-fit your source to the canvas
Add _saccade.py as a script

# Other stuff
There's a bunch of hard-coded variables at the top of the script, stuff like Canvas geometry, framerate, and panning strength are tweakable.

I don't know, good luck.
