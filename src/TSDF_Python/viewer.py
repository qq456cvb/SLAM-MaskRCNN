from OpenGL import GL as gl
from OpenGL.GL import shaders
import ctypes
import sys
import numpy as np


class Viewer:
    def __init__(self):
        # Load Shader
        self.attrib_locs = {
            "aPosCoord": -1,
            "aTexCoord": -1
        }
        self.uniform_locs = {
            "tsdf": -1,
            "tsdf_cnt": -1,
            "s2w": -1,
            "c": -1,
            "vol_dim": -1,
            "volStart": -1,
            "volEnd": -1,
            "random_colors": -1
        }
        vert_prog = shaders.compileShader(open('tsdf_render.vert').read(), gl.GL_VERTEX_SHADER)
        if not gl.glGetShaderiv(vert_prog, gl.GL_COMPILE_STATUS):
            sys.stderr.write("Error: Could not compile vertex shader.\n")
            exit(2)
        frag_prog = shaders.compileShader(open('tsdf_render.frag').read(), gl.GL_FRAGMENT_SHADER)
        if not gl.glGetShaderiv(frag_prog, gl.GL_COMPILE_STATUS):
            sys.stderr.write("Error: Could not compile fragment shader.\n")
            exit(3)
        self.program = gl.glCreateProgram()
        gl.glAttachShader(self.program, vert_prog)
        gl.glAttachShader(self.program, frag_prog)
        gl.glLinkProgram(self.program)
        if gl.glGetProgramiv(self.program, gl.GL_LINK_STATUS) != gl.GL_TRUE:
            sys.stderr.write("Error: {0}\n".format(gl.glGetProgramInfoLog(self.program)))
            exit(4)
        for name in self.attrib_locs:
            self.attrib_locs[name] = gl.glGetAttribLocation(self.program, name)
        for name in self.uniform_locs:
            self.uniform_locs[name] = gl.glGetUniformLocation(self.program, name)

        vertices = np.array([-1, -1, -1, 1, 1, -1, 1, 1], np.float32)
        texture_coords = np.array([0, 1, 0, 0, 1, 1, 1, 0], np.float32)

        # Load Object
        self.vbos = gl.glGenBuffers(2)
        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

        gl.glEnableVertexAttribArray(self.attrib_locs['aPosCoord'])
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbos[0])
        gl.glBufferData(gl.GL_ARRAY_BUFFER, 8 * 4, vertices, gl.GL_STATIC_DRAW)
        gl.glVertexAttribPointer(self.attrib_locs['aPosCoord'], 2, gl.GL_FLOAT, False, 0, ctypes.c_void_p(0))
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

        gl.glEnableVertexAttribArray(self.attrib_locs['aTexCoord'])
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbos[1])
        gl.glBufferData(gl.GL_ARRAY_BUFFER, 8 * 4, texture_coords, gl.GL_STATIC_DRAW)
        gl.glVertexAttribPointer(self.attrib_locs['aTexCoord'], 2, gl.GL_FLOAT, False, 0, ctypes.c_void_p(0))
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

        gl.glBindVertexArray(0)

        gl.glUseProgram(self.program)
        gl.glUniform1i(self.uniform_locs['tsdf'], 0)
        gl.glUniform1i(self.uniform_locs['tsdf_cnt'], 1)

        rand_colors = np.random.rand(32, 3)
        gl.glUniform3fv(self.uniform_locs['random_colors'], 32, rand_colors.astype(np.float32))

    def set_s2w(self, mat):
        gl.glUseProgram(self.program)
        gl.glUniformMatrix4fv(self.uniform_locs['s2w'], 1, True, mat.astype(np.float32))

    def set_c(self, c):
        gl.glUseProgram(self.program)
        gl.glUniform3fv(self.uniform_locs['c'], 1, c.astype(np.float32))

    def set_vol_dim(self, vol_dim):
        gl.glUseProgram(self.program)
        gl.glUniform1f(self.uniform_locs['vol_dim'], np.float32(vol_dim))

    def set_vol_start(self, vol_start):
        gl.glUseProgram(self.program)
        gl.glUniform3fv(self.uniform_locs['volStart'], 1, vol_start.astype(np.float32))

    def set_vol_end(self, vol_end):
        gl.glUseProgram(self.program)
        gl.glUniform3fv(self.uniform_locs['volEnd'], 1, vol_end.astype(np.float32))