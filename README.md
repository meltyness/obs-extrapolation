# obs-extrapolation
Saccade is a weird OBS Python script that is weird

`Saccade makes recordings of perspective / widescreen games display more salient information when packed into portrait-form content. 
It can do this in real-time for streaming.`

It is primarily based on https://github.com/upgradeQ/OBS-Studio-Python-Scripting-Cheatsheet-obspython-Examples-of-API/blob/master/src/get_source_frame_data_ffi.py

It sort of works, but it applies smoothed proportional control correlated with scene motion and some other hacky control stuff.

It badly needs to be refactored and it also crashes unpredictably under various circumstances that I can't explain.

