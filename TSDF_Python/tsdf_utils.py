import numpy as np
import math
import cv2
import sdl2
import sys
from OpenGL import GL as gl
from OpenGL.GL import shaders
from TSDF_Python.viewer import Viewer
import ctypes
import math


def filter_gaussian(img):
    threshold = 3
    if np.sum(img) == 0:
        return img, 0
    mean = np.mean(img[img > 0])
    std = np.std(img[img > 0])
    img[abs(img-mean) > threshold * std] = 0
    return img, np.mean(img[img > 0])


def read_traj(filename):
    with open(filename) as f:
        lines = f.readlines()
    traj = np.array([l.strip('\n').split(' ') for l in lines if not l.startswith('#')])
    traj[:, 0] = [t[5:] for t in traj[:, 0]]
    traj = traj.astype(np.double)
    return traj


def transform44(l):
    """
    Generate a 4x4 homogeneous transformation matrix from a 3D point and unit quaternion.

    Input:
    l -- tuple consisting of (stamp,tx,ty,tz,qx,qy,qz,qw) where
         (tx,ty,tz) is the 3D position and (qx,qy,qz,qw) is the unit quaternion.

    Output:
    matrix -- 4x4 homogeneous transformation matrix
    """
    t = l[:3]
    q = np.array(l[3:7], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < 1e-7:
        return np.array([
            (1.0, 0.0, 0.0, t[0]),
            (0.0, 1.0, 0.0, t[1]),
            (0.0, 0.0, 1.0, t[2]),
            (0.0, 0.0, 0.0, 1.0)
        ], dtype=np.float64)
    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)
    mat = np.array((
        (1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3], t[0]),
        (q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3], t[1]),
        (q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1], t[2]),
        (0.0, 0.0, 0.0, 1.0)
    ), dtype=np.float64)
    return np.linalg.inv(mat)


def parse_pos(pos):
    axis = pos[3:-1]

    theta = 2 * math.atan2(np.linalg.norm(axis), pos[-1])
    axis = axis / np.linalg.norm(axis)

    rod = theta * axis
    rot = np.zeros([3, 3])
    cv2.Rodrigues(rod, rot)

    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rot
    extrinsic[:-1, -1] = pos[:3]
    return np.linalg.inv(extrinsic)


def slerp(q1, q2, t):
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    dot = np.dot(q1, q2)

    if dot < 0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result

    dot = max(min(dot, 1), -1)
    theta_0 = math.acos(dot)
    theta = theta_0 * t

    s1 = math.cos(theta) - dot * math.sin(theta) / math.sin(theta_0)
    s2 = math.sin(theta) / math.sin(theta_0)
    return s1 * q1 + s2 * q2


def fix_distortion(img, intrinsic, dist):
    return cv2.undistort(img, intrinsic, dist)


def show_model(tsdf):

    WINDOW_WIDTH = 640
    WINDOW_HEIGHT = 480
    # Init
    sdl2.SDL_Init(sdl2.SDL_INIT_EVERYTHING)
    sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_MAJOR_VERSION, 3)
    sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_MINOR_VERSION, 2)
    sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_PROFILE_MASK,
                             sdl2.SDL_GL_CONTEXT_PROFILE_CORE)
    sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_DOUBLEBUFFER, 1)
    sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_DEPTH_SIZE, 24)
    sdl2.SDL_GL_SetSwapInterval(1)
    window = sdl2.SDL_CreateWindow(
        b"Python/SDL2/OpenGL", sdl2.SDL_WINDOWPOS_CENTERED,
        sdl2.SDL_WINDOWPOS_CENTERED, WINDOW_WIDTH, WINDOW_HEIGHT,
        sdl2.SDL_WINDOW_OPENGL | sdl2.SDL_WINDOW_SHOWN)
    if not window:
        sys.stderr.write("Error: Could not create window\n")
        exit(1)
    glcontext = sdl2.SDL_GL_CreateContext(window)
    # gl.glClampColor(gl.GL_CLAMP_FRAGMENT_COLOR, False)
    # gl.glClampColor(gl.GL_CLAMP_VERTEX_COLOR, False)
    # gl.glClampColor(gl.GL_CLAMP_READ_COLOR, False)

    viewer = Viewer()
    viewer.set_s2w(tsdf.intrinsic_inv)
    viewer.set_c(np.zeros([3]))
    viewer.set_vol_dim(tsdf.vol_dim)
    viewer.set_vol_start(tsdf.vol_start)
    viewer.set_vol_end(tsdf.vol_end)

    tex = gl.glGenTextures(2)
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex[0])

    # print(tsdf.tsdf_color.dtype)
    fused = np.concatenate([tsdf.tsdf_color.astype(np.float32) / 255, np.expand_dims(tsdf.tsdf_diff, -1)], axis=-1)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, fused.shape[1], fused.shape[0], 0, gl.GL_RGBA, gl.GL_FLOAT, fused)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    gl.glBindTexture(gl.GL_TEXTURE_2D, tex[1])
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_R16I, tsdf.tsdf_cls.shape[1], tsdf.tsdf_cls.shape[0], 0, gl.GL_RED_INTEGER, gl.GL_INT,
                    tsdf.tsdf_cls)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    running = True
    event = sdl2.SDL_Event()
    angle = 0
    while running:
        angle += 0.01
        while sdl2.SDL_PollEvent(ctypes.byref(event)) != 0:
            if event.type == sdl2.SDL_QUIT:
                running = False
            if event.type == sdl2.events.SDL_KEYDOWN:
                print("SDL_KEYDOWN")
                if event.key.keysym.sym == sdl2.SDLK_ESCAPE:
                    running = False
            # if event.type == sdl2.SDL_MOUSEMOTION:
            #     print("SDL_MOUSEMOTION")
            # if event.type == sdl2.SDL_MOUSEBUTTONDOWN:
            #     print("SDL_MOUSEBUTTONDOWN")
        mean_depth = tsdf.mean_depth
        rot = np.array([[math.cos(angle), 0, -math.sin(angle), mean_depth * math.sin(angle)],
                        [0, 1, 0, 0],
                        [math.sin(angle), 0, math.cos(angle), mean_depth - mean_depth * math.cos(angle)],
                        [0, 0, 0, 1]])
        viewer.set_s2w(np.matmul(rot, tsdf.intrinsic_inv))
        viewer.set_c(np.array([(mean_depth + 0.5) * math.sin(angle), 0, (mean_depth + 0.5) - (mean_depth + 0.5) * math.cos(angle)]))

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
        gl.glUseProgram(viewer.program)

        gl.glBindVertexArray(viewer.vao)

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex[0])

        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex[1])

        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
        # gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glUseProgram(0)
        sdl2.SDL_GL_SwapWindow(window)
