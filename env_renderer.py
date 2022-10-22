# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import render_module as rm
from fairmotion.viz import glut_viewer
from fairmotion.viz.utils import TimeChecker
tc = TimeChecker()

from collections import deque
import pickle

rm.initialize()

def axis_to_str(axis):
    if np.allclose(axis, np.array([1.0, 0.0, 0.0])):
        return 'x'
    elif np.allclose(axis, np.array([0.0, 1.0, 0.0])):
        return 'y'
    elif np.allclose(axis, np.array([0.0, 0.0, 1.0])):
        return 'z'
    else: 
        raise Exception

class EnvRenderer(glut_viewer.Viewer):
    def __init__(self,
                 env=None,
                 title="env_renderer",
                 cam=None,
                 size=(1280, 720),
                 renderer='inhouse',
                 bgcolor=[0.0, 0.0, 0.0, 1.0],
                 ground_tex_file="data/image/grid2.png"):
        super().__init__(title, cam, size, bgcolor)
        self.rm = rm
        self.env = env
        self.render_return_times = deque(maxlen=10)
        self.ground_tex_file = ground_tex_file
        assert renderer in ['inhouse', 'bullet_native']
        self.renderer = renderer
        self.options = {
            "ground_color": [0.8,0.8,0.8,1.0],
            "ground_size": [40.0, 40.0],
            "ground_dsize": [2.0, 2.0],
            "ground_circle_cut": True,
            "draw_underlying_mesh": False,
            "mesh_line_scale": [1.0, 1.0, 1.0],
            "mesh_line_width": 1.0,
            "mesh_line_color": [0.0, 0.0, 0.0, 1.0],
        }

        if self.renderer == 'bullet_native':
            self.tex_overlay = None
    def use_default_ground(self):
        raise NotImplementedError
    def get_v_up_env_str(self):
        raise NotImplementedError
    def get_pb_client(self):
        raise NotImplementedError
    def get_ground(self):
        raise NotImplementedError
    def render_ground(self):
        if self.use_default_ground():
            self.render_ground_defulat(
                axis=self.get_v_up_env_str(),
                size=self.options['ground_size'],
                dsize=self.options['ground_dsize'], 
                circle_cut=self.options['ground_circle_cut'])
        else:
            self.rm.bullet_render.render_model(
                self.get_pb_client(), 
                self.get_ground(),
                draw_link=True, 
                draw_link_info=self.options["draw_underlying_mesh"], 
                draw_joint=False, 
                draw_joint_geom=False, 
                link_info_scale=self.options["mesh_line_scale"], 
                link_info_color=self.options["mesh_line_color"], 
                link_info_line_width=self.options["mesh_line_width"], 
                color=self.options["ground_color"], 
                lighting=True)
    def render_ground_defulat(self, axis, size, dsize, circle_cut):
        if self.rm.tex_id_ground is None:
            self.rm.tex_id_ground = \
              self.rm.gl_render.load_texture(self.ground_tex_file)
        self.rm.gl_render.render_ground_texture(
            self.rm.tex_id_ground,
            size=size,
            dsize=dsize,
            axis=axis,
            origin=self.rm.flag['origin'],
            use_arrow=True,
            circle_cut=circle_cut)
    def extra_keyboard_callback(self, key):
        pass
    def extra_render_callback(self):
        pass
    def extra_idle_callback(self):
        pass
    def extra_overlay_callback(self):
        pass
    def keyboard_callback(self, key):
        if key in self.rm.toggle:
            keyword = self.rm.toggle[key]
            flag = self.rm.flag[keyword]
            if isinstance(flag, list):
                flag[0] = (flag[0]+1)%len(flag[1])
                print('Reserved Flag<%s>:'%key, keyword, flag[1][flag[0]])
            elif isinstance(flag, bool):
                self.rm.flag[keyword] = not flag
                print('Reserved Key<%s>:'%key, keyword, self.rm.flag[keyword])
            else:
                raise NotImplementedError
        elif key == b'M':
                filename = 'data/temp/temp.cam'
                with open(filename, "wb") as file:
                    pickle.dump(self.cam_cur, file)
                    print('Saved:', filename)
        elif key == b'm':
            filename = 'data/temp/temp.cam'
            with open(filename, "rb") as file:
                self.cam_cur = pickle.load(file)
                print('Loaded:', filename)
        else:
            self.extra_keyboard_callback(key)
    def render_callback(self):
        if self.renderer == 'inhouse':
            if not self.rm.flag['all_scene']: return
            self.render_return_times.append(tc.get_time())
            self.update_cam()
            if self.rm.flag['ground']:
                self.render_ground()
            self.extra_render_callback()
    def idle_callback(self):
        self.extra_idle_callback()
    def overlay_callback(self):
        gl = self.rm.gl
        if self.renderer == 'bullet_native':
            pb_client = self.get_pb_client()
            ''' Get a rendered image from bullet API '''
            view_matrix = pb_client.computeViewMatrix(
                self.cam_cur.pos, 
                self.cam_cur.origin, 
                self.cam_cur.vup,
            )
            proj_matrix = pb_client.computeProjectionMatrixFOV(
                self.cam_cur.fov, 
                self.window_size[0]/self.window_size[1], 
                0.1, 
                50,
            )
            w, h, rgba, depth, mask = pb_client.getCameraImage(
                width=self.window_size[0],
                height=self.window_size[1],
                projectionMatrix=proj_matrix,
                viewMatrix=view_matrix,
                renderer=pb_client.ER_TINY_RENDERER,
                )

            ''' Copy the rendered image to the overlay texture '''

            if self.tex_overlay is None:
                self.tex_overlay = gl.glGenTextures(1)

            gl.glEnable(gl.GL_TEXTURE_2D)

            gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex_overlay)
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D, 0, gl.GL_RGB, w, h, 0, 
                gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, rgba
            )
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
            gl.glTexEnvf(gl.GL_TEXTURE_ENV, gl.GL_TEXTURE_ENV_MODE, gl.GL_DECAL)

            ''' Render the texture '''
            
            gl.glBegin(rm.gl.GL_QUADS)
            
            gl.glTexCoord2d(0.0, 0.0)
            gl.glVertex2d(0.0, 0.0)
            
            gl.glTexCoord2d(1.0, 0.0)
            gl.glVertex2d(w, 0.0)
            
            gl.glTexCoord2d(1.0, 1.0)
            gl.glVertex2d(w, h)
            
            gl.glTexCoord2d(0.0, 1.0)
            gl.glVertex2d(0.0, h)
            
            gl.glEnd()

            gl.glDisable(gl.GL_TEXTURE_2D)
            
        if not self.rm.flag['overlay']: return
        
        gl.glPushAttrib(gl.GL_LIGHTING)
        gl.glDisable(gl.GL_LIGHTING)
        
        self.extra_overlay_callback()

        if self.rm.flag['overlay_text']:
            w, h = self.window_size
            self.rm.gl_render.render_text(
                "FPS: %.2f"%self.get_avg_fps(), 
                pos=[0.05*w, 0.9*h], 
                font=self.rm.glut.GLUT_BITMAP_9_BY_15)
            self.rm.gl_render.render_text(
                "Time: %.2f"%self.get_elapsed_time(), 
                pos=[0.05*w, 0.9*h+20], 
                font=self.rm.glut.GLUT_BITMAP_9_BY_15)
            self.rm.gl_render.render_text(
                "Size: %d x %d"%(self.window_size[0],self.window_size[1]),
                pos=[0.05*w, 0.9*h+40], 
                font=self.rm.glut.GLUT_BITMAP_9_BY_15)

        gl.glPopAttrib(gl.GL_LIGHTING)
    def get_avg_fps(self):
        if len(self.render_return_times) > 0:
            return int(1.0/np.mean(self.render_return_times))
        else:
            return 0
    def update_cam(self):
        if self.rm.get_flag('follow_cam') != "fixed":
            param = self.get_cam_parameters()
            if param.get('origin') is not None:
                self.cam_cur.origin = param['origin']
            if param.get('pos') is not None:
                self.cam_cur.pos = param['pos']
            if param.get('dist') is not None:
                self._update_cam_distance(param['dist'])
            if param.get('translate') is not None:
                self._update_cam_translate(
                    param['translate']['target_pos'],
                    param['translate'].get('ignore_x'),
                    param['translate'].get('ignore_y'),
                    param['translate'].get('ignore_z'))

    def _update_cam_translate(
        self, target_pos, ignore_x=False, ignore_y=False, ignore_z=False):
        if np.array_equal(target_pos, self.cam_cur.origin):
            return
        d = target_pos - self.cam_cur.origin
        if ignore_x: d[0] = 0.0
        if ignore_y: d[1] = 0.0
        if ignore_z: d[2] = 0.0
        self.cam_cur.translate(d)
    def _update_cam_distance(self, d):
        assert d > 0.0
        cam = self.cam_cur
        vl = cam.pos - cam.origin
        length = np.linalg.norm(vl)
        cam.pos = cam.origin + d * (vl / length)
    def get_cam_parameters(self):
        return {
            'origin': None, 
            'pos': None, 
            'dist': None,
            'translate': None,
        }
    def get_elapsed_time(self):
        return 0.0
