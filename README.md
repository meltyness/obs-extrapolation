# obs-extrapolation
Saccade is a script that is weird

It is primarily based on https://github.com/upgradeQ/OBS-Studio-Python-Scripting-Cheatsheet-obspython-Examples-of-API/blob/master/src/get_source_frame_data_ffi.py

It sort of works, but it applies smoothed proportional control correlated with scene motion and some other hacky control stuff.

It badly needs to be refactored and it also crashes unpredictably under various circumstances that I can't explain.

The basic idea is to automatically make perspective / widescreen games display more salient information when packed into portrait-form content.
