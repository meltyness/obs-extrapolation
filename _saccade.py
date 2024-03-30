from types import SimpleNamespace as _G
from ctypes import *
from ctypes.util import find_library
import obspython as obs

import collections
import math
import time
import cv2
import numpy as np

peeky, ffi = _G(), _G()

ffi.obsffi = CDLL(find_library("obs"))  # ? PyDLL

peeky.lock = False
peeky.start_delay = 2
peeky.duration = 0
peeky.noise = 999
peeky.tick = 4
peeky.tick_mili = peeky.tick * 0.001
peeky.interval_sec = 0.01605
peeky.tick_acc = 0
peeky.kp = None
peeky.kp_last = None
peeky.des = None
peeky.des_last = None

peeky.source_name = "Spaceship"

peeky.init_vec = obs.vec2()
peeky.disturbance = obs.vec2()

peeky.current_scene_as_source = obs.obs_frontend_get_current_scene()
peeky.scene_item = None
peeky.current_scene = None
if peeky.current_scene_as_source:
    peeky.current_scene = obs.obs_scene_from_source(peeky.current_scene_as_source)

    peeky.scene_item = obs.obs_scene_find_source_recursive(peeky.current_scene, peeky.source_name)

    if peeky.scene_item:
        obs.obs_sceneitem_get_pos(peeky.scene_item, peeky.init_vec)
        obs.obs_sceneitem_get_pos(peeky.scene_item, peeky.disturbance)

peeky.init_vec.x -= 740

obs.obs_sceneitem_release(peeky.scene_item)
obs.obs_scene_release(peeky.current_scene)
obs.obs_source_release(peeky.current_scene_as_source)

peeky.render_texture = obs.gs_texrender_create(obs.GS_RGBA, obs.GS_ZS_NONE)
peeky.rgba_4b = 4  # bytes
peeky.surface = None

peeky.jump_limiter = 0

peeky.jump_min = 48

peeky.jump_max = 128
peeky.urgency = 3 * peeky.jump_max

peeky.orb = cv2.ORB_create()
peeky.orb.setMaxFeatures(60)
# Probably FAST idk there's no docs that i watn to reead
peeky.orb.setScoreType(1)


peeky.proportional_control = 9
peeky.push_scale = 16

peeky.buffered_stylus = collections.deque([peeky.init_vec.x], maxlen=7)


# cv2.namedWindow("Debug Output")
# cv2.waitKey(1)
print("loaded ffi script")

# def script_tick(seconds):
#     if (not peeky.kp_last is None) and (not peeky.des is None):
#         print("moving spaceship")
#         current_scene_as_source = obs.obs_frontend_get_current_scene()
#         print("or nah")
#         scene_item = None
#         current_scene = None
#         if current_scene_as_source:
#             current_scene = obs.obs_scene_from_source(current_scene_as_source)

#             scene_item = obs.obs_scene_find_source_recursive(current_scene, peeky.source_name)

#             obs.obs_sceneitem_set_pos(scene_item, peeky.disturbance)

#         obs.obs_sceneitem_release(scene_item)
#         obs.obs_scene_release(current_scene)
#         obs.obs_source_release(current_scene_as_source)

def script_tick(seconds):
    current_scene_as_source = obs.obs_frontend_get_current_scene()

    if current_scene_as_source:
        current_scene = obs.obs_scene_from_source(current_scene_as_source)
        scene_item = obs.obs_scene_find_source_recursive(current_scene, peeky.source_name)

        if scene_item:
            peeky.buffered_stylus.append(peeky.disturbance.x)
            draw_vec = obs.vec2()
            draw_vec.y = peeky.disturbance.y
            draw_vec.x = np.mean(peeky.buffered_stylus)
            obs.obs_sceneitem_set_pos(scene_item, draw_vec)

    obs.obs_source_release(current_scene_as_source)

def wrap(funcname, restype, argtypes=None, use_lib=None):
    """Simplify wrapping ctypes functions"""
    if use_lib is not None:
        func = getattr(use_lib, funcname)
    else:
        func = getattr(ffi.obsffi, funcname)
    func.restype = restype
    if argtypes is not None:
        func.argtypes = argtypes
    ffi.__dict__[funcname] = func


class TexRender(Structure):
    pass


class StageSurf(Structure):
    pass


wrap("gs_stage_texture", None, argtypes=[POINTER(StageSurf), POINTER(TexRender)])
wrap("gs_stagesurface_create", POINTER(StageSurf), argtypes=[c_uint, c_uint, c_int])
wrap(
    "gs_stagesurface_map",
    c_bool,
    argtypes=[POINTER(StageSurf), POINTER(POINTER(c_ubyte)), POINTER(c_uint)],
)
wrap("gs_stagesurface_destroy", None, argtypes=[POINTER(StageSurf)])
wrap("gs_stagesurface_unmap", None, argtypes=[POINTER(StageSurf)])


def output_to_stdout():
    obs.obs_enter_graphics()
    source = obs.obs_get_source_by_name(peeky.source_name)
    # print(dir(source))
    if source and obs.gs_texrender_begin(peeky.render_texture, 1920, 1080):
        obs.obs_source_video_render(source)
        obs.gs_texrender_end(peeky.render_texture)
        if not peeky.surface:
            peeky.surface = ffi.gs_stagesurface_create(
                c_uint(1920), c_uint(1080), c_int(obs.GS_RGBA)
            )
        tex = obs.gs_texrender_get_texture(peeky.render_texture)
        tex = c_void_p(int(tex))
        tex = cast(tex, POINTER(TexRender))
        ffi.gs_stage_texture(peeky.surface, tex)
        data = POINTER(c_ubyte)()
        if ffi.gs_stagesurface_map(peeky.surface, byref(data), byref(c_uint(peeky.rgba_4b))):
            w = obs.obs_source_get_width(source)
            h = obs.obs_source_get_height(source)       
            manage_imgproc(data, w, h)
            ffi.gs_stagesurface_unmap(peeky.surface)
        
        obs.gs_texrender_reset(peeky.render_texture)
    obs.obs_source_release(source)
    obs.obs_leave_graphics()

peeky.callback = output_to_stdout

def manage_imgproc(data, w, h):
    # Convert FFI Byte Array to Numpy
    arr = np.ctypeslib.as_array(data, shape=(1920*1080*4,)).reshape(1080, 1920, 4)

    # Crop to known image size
    peeky.arr = arr[0:h, 0:w, [2, 1, 0, 3]]

    # Convert to Grayscale for Feature Detect step
    gray_img = cv2.cvtColor(peeky.arr, cv2.COLOR_BGR2GRAY)
    
    gray_img = cv2.resize(gray_img, (240, 135), 
               interpolation = cv2.INTER_LINEAR)
    # Record previous orb run
    if not peeky.kp is None:
        peeky.kp_last = peeky.kp
        peeky.des_last = peeky.des


    peeky.kp, peeky.des = peeky.orb.detectAndCompute(gray_img,None)

    if not peeky.des is None:
        peeky.des = peeky.des.astype(np.float32)

    # Display thedebugger p
    #cv2.namedWindow('Debug Output', cv2.WINDOW_FREERATIO | cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_EXPANDED)
    #cv2.imshow('Debug Output', peeky.arr)
    #cv2.pollKey()

    if (not peeky.kp_last is None) and (not peeky.des is None) and (not peeky.des_last is None) and (len(peeky.des) > 1 and len(peeky.des_last) > 1):
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50) # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        # print(f"Length of peeky.des {len(peeky.des)} and des_last {len(peeky.des_last)}")
        matches = flann.knnMatch(peeky.des,peeky.des_last,k=2)
        # Initialize an empty list to store good matches
        good_matches = []

        # Iterate over the matches
        for m, n in matches:
            # Check if the distance ratio is less than a threshold (e.g., 0.75)
            if m.distance < 0.9 * n.distance:
                good_matches.append(m)

        # Initialize variables to store the offsets
        dx = 0
        dy = 0

        # Iterate over the good matches
        for match in good_matches:
            # Get the coordinates of the matched keypoints
            pt_a = peeky.kp[match.queryIdx].pt
            pt_b = peeky.kp_last[match.trainIdx].pt

            # Calculate the offsets
            dx += pt_a[0] - pt_b[0]
            dy += pt_a[1] - pt_b[1]

        # Calculate the average offsets
        if len(good_matches) > 0:
            dx /= len(good_matches)
            dy /= len(good_matches)
        else:
            dx = 0
            dy = 0
        # print(f"dx {dx} dy {dy}")                
        # print(f"peeky disturbance: {peeky.disturbance.x}, limiter status: {peeky.jump_limiter}")

        # Apply bumps
        if(dy < 0.05):
            increment = dx * peeky.push_scale
        else:
            increment = 0

        # Limiter kicks in
        if abs(increment) > peeky.jump_max:
            if increment > 0:
                # Increase limiter buffer by overflowed amount
                peeky.jump_limiter += increment - peeky.jump_max
            else:
                # Decrement limiter buffer by overflowed amount (it's a negative increment)
                peeky.jump_limiter += increment + peeky.jump_max
        elif abs(increment) > peeky.jump_min:
            # Limiter behavior not needed also means that abs()
            peeky.disturbance.x += increment

        # While we have accumulated jumps, bleed them off in (at most) jump_max increments
        moveamt = min(abs(peeky.jump_limiter), abs(peeky.jump_max))
        if peeky.jump_limiter > 0:
            peeky.disturbance.x += moveamt
            peeky.jump_limiter -= moveamt
        else:
            peeky.disturbance.x -= moveamt
            peeky.jump_limiter += moveamt

        # Limit at edges
        if peeky.disturbance.x < -1600:
            peeky.disturbance.x = -1600

        if peeky.disturbance.x > 0:
            peeky.disturbance.x = 0

        # Stop near the center
        if abs(peeky.disturbance.x - peeky.init_vec.x) > peeky.jump_min:
            peeky.disturbance.x -= (np.cbrt((peeky.disturbance.x - peeky.init_vec.x)) * peeky.proportional_control)

def event_loop():
    """wait n seconds, then execute callback within certain interval"""
    if peeky.duration > peeky.start_delay:
        peeky.tick_acc += peeky.tick_mili
        if peeky.tick_acc > peeky.interval_sec:
            peeky.callback()
            peeky.tick_acc = 0
            return
    else:
        peeky.duration += peeky.tick_mili

obs.timer_add(event_loop, peeky.tick)

def script_unload():
    "clean up"
    obs.obs_enter_graphics()
    ffi.gs_stagesurface_destroy(peeky.surface)
    obs.gs_texrender_destroy(peeky.render_texture)
    obs.obs_leave_graphics()
