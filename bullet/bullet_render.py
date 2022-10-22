# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np
import pybullet
from fairmotion.viz import gl_render
from fairmotion.ops import conversions
from fairmotion.utils import constants

mesh_database = {}

VERTEX_FORMATS = {
    'V3F': GL_V3F,
    'C3F_V3F': GL_C3F_V3F,
    'N3F_V3F': GL_N3F_V3F,
    'T2F_V3F': GL_T2F_V3F,
    # 'C3F_N3F_V3F': GL_C3F_N3F_V3F,  # Unsupported
    'T2F_C3F_V3F': GL_T2F_C3F_V3F,
    'T2F_N3F_V3F': GL_T2F_N3F_V3F,
    # 'T2F_C3F_N3F_V3F': GL_T2F_C3F_N3F_V3F,  # Unsupported
}

def render_meshes(obj, lighting_enabled=True, textures_enabled=True):
    materials = obj.materials
    """Draw a dict of meshes"""
    for name, material in materials.items():
        render_mesh(material, lighting_enabled=lighting_enabled, textures_enabled=textures_enabled)

def render_mesh(material, face=GL_FRONT_AND_BACK, lighting_enabled=True, textures_enabled=True):
    """Draw a single material"""
    if material.gl_floats is None:
        material.gl_floats = (GLfloat * len(material.vertices))(*material.vertices)
        material.triangle_count = len(material.vertices) / material.vertex_size

    vertex_format = VERTEX_FORMATS.get(material.vertex_format)
    if not vertex_format:
        raise ValueError("Vertex format {} not supported by pyglet".format(material.vertex_format))

    glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT)
    glPushAttrib(GL_CURRENT_BIT | GL_ENABLE_BIT | GL_LIGHTING_BIT)
    glEnable(GL_CULL_FACE)
    glEnable(GL_DEPTH_TEST)
    glCullFace(GL_BACK)

    # if textures_enabled:
    #     # Fall back to ambient texture if no diffuse
    #     texture = material.texture or material.texture_ambient
    #     if texture and material.has_uvs:
    #         bind_texture(texture)
    #     else:
    #         glDisable(GL_TEXTURE_2D)

    
    # if lighting_enabled and material.has_normals:
    #     glMaterialfv(face, GL_DIFFUSE, gl_light(material.diffuse))
    #     glMaterialfv(face, GL_AMBIENT, gl_light(material.ambient))
    #     glMaterialfv(face, GL_SPECULAR, gl_light(material.specular))
    #     glMaterialfv(face, GL_EMISSION, gl_light(material.emissive))
    #     glMaterialf(face, GL_SHININESS, min(128.0, material.shininess))
    #     glEnable(GL_LIGHT0)
    #     glEnable(GL_LIGHTING)
    # else:
    #     glDisable(GL_LIGHTING)
    #     glColor4f(*material.ambient)

    glEnable(GL_LIGHTING)

    glInterleavedArrays(vertex_format, 0, material.gl_floats)
    glDrawArrays(GL_TRIANGLES, 0, int(material.triangle_count))

    glPopAttrib()
    glPopClientAttrib()

def render_geom_bounding_box(geom_type, geom_size, color=[0, 0, 0, 1], T=constants.EYE_T):
    if geom_type==pybullet.GEOM_SPHERE:
        size = [2*geom_size[0], 2*geom_size[0], 2*geom_size[0]]
        gl_render.render_cube(T, size=size, color=color, solid=False)
    elif geom_type==pybullet.GEOM_BOX:
        size = [geom_size[0], geom_size[1], geom_size[2]]
        gl_render.render_cube(T, size=size, color=color, solid=False)
    elif geom_type==pybullet.GEOM_CAPSULE:
        size = [2*geom_size[1], 2*geom_size[1], 2*geom_size[1]+geom_size[0]]
        gl_render.render_cube(T, size=size, color=color, solid=False)
    else:
        raise NotImplementedError()

# def render_geom(geom_type, geom_size, color=[0.5, 0.5, 0.5, 1.0], T=constants.EYE_T):
def render_geom(data_visual, color=None, T=None):
    type = data_visual[2]
    param = data_visual[3]
    if T is None:
        T = conversions.Qp2T(np.array(data_visual[6]), np.array(data_visual[5]))
    else:
        T = constants.EYE_T
    glPushMatrix()
    gl_render.glTransform(T)
    if color is None:
        color = data_visual[7]
    if type==pybullet.GEOM_SPHERE:
        gl_render.render_sphere(
            constants.EYE_T, param[0], color=color, slice1=16, slice2=16)
    elif type==pybullet.GEOM_BOX:
        gl_render.render_cube(
            constants.EYE_T, size=[param[0], param[1], param[2]], color=color)
    elif type==pybullet.GEOM_CAPSULE:
        gl_render.render_capsule(
            constants.EYE_T, param[0], param[1], color=color, slice=16)
    elif type==pybullet.GEOM_CYLINDER:
        gl_render.render_cylinder(
            constants.EYE_T, param[0], param[1], color=color, slice=16)
    # elif type==pybullet.GEOM_PLANE:
    #     pass
    elif type==pybullet.GEOM_MESH:
        import pywavefront
        from pywavefront import visualization
        filename = data_visual[4].decode("UTF-8")
        if filename in mesh_database.keys():
            obj = mesh_database[filename]
        else:
            ext = os.path.splitext(filename)[-1].lower()
            if not ext=='.obj':
                raise NotImplementedError
            obj = pywavefront.Wavefront(filename)
            mesh_database[filename] = obj
        gl_render.glColor(color)
        visualization.draw(obj)
    else:
        raise NotImplementedError()
    glPopMatrix()

# def render_geom_info(geom_type, geom_size, scale=1.0, color=[0, 0, 0, 1], T=constants.EYE_T, line_width=1.0, num_slice=32):
def render_geom_info(data_visual, color=None, scale=1.0, T=None, line_width=1.0, num_slice=32):
    type = data_visual[2]
    param = data_visual[3]
    if T is None:
        T = conversions.Qp2T(np.array(data_visual[6]), np.array(data_visual[5]))
    else:
        T = constants.EYE_T
    if color is None:
        color = [0.2, 0.2, 0.2, 1]
    glLineWidth(line_width)
    glPushMatrix()
    gl_render.glTransform(T)
    if isinstance(scale, list):
        glScalef(scale[0], scale[1], scale[2])
    else:
        glScalef(scale, scale, scale)
    if type==pybullet.GEOM_SPHERE:
        gl_render.render_sphere_info(
            constants.EYE_T, param[0], line_width=line_width, slice=num_slice,  color=color)
    elif type==pybullet.GEOM_BOX:
        gl_render.render_cube(
            constants.EYE_T, size=[param[0], param[1], param[2]], color=color, solid=False, line_width=line_width)
    elif type==pybullet.GEOM_CAPSULE:
        gl_render.render_capsule_info(
            constants.EYE_T, param[0], param[1], line_width=line_width, slice=num_slice, color=color)
    elif type==pybullet.GEOM_CYLINDER:
        gl_render.render_cylinder_info(
            constants.EYE_T, param[0], param[1], line_width=line_width, slice=num_slice, color=color)
    # elif type==pybullet.GEOM_PLANE:
    #     pass
    elif type==pybullet.GEOM_MESH:
        import pywavefront
        from pywavefront import visualization
        filename = data_visual[4].decode("UTF-8")
        if filename in mesh_database.keys():
            obj = mesh_database[filename]
        else:
            ext = os.path.splitext(filename)[-1].lower()
            if not ext=='.obj':
                raise NotImplementedError
            obj = pywavefront.Wavefront(filename)
            mesh_database[filename] = obj
        gl_render.glColor(color)
        glDisable(GL_LIGHTING)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        visualization.draw(obj)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glEnable(GL_LIGHTING)
    else:
        raise NotImplementedError()
    glPopMatrix()

def render_links_pos(pb_client, model):
    for j in range(pb_client.getNumJoints(model)):
        link_state = pb_client.getLinkState(model, j)
        p, Q = np.array(link_state[0]), np.array(link_state[1])
        T = conversions.Qp2T(Q, p)
        gl_render.render_point(p, radius=0.01, color=[0, 1, 0, 1])

def render_links(
    pb_client,
    model,
    color=None,
    draw_link=True, 
    draw_link_info=True,  
    link_info_scale=1.0,
    link_info_color=[0, 0, 0, 1],
    link_info_line_width=1.0,
    link_info_num_slice=32,
    link_filter=None,
    lighting=True,
    ):
    data_visual = pb_client.getVisualShapeData(model)

    if lighting:
        glEnable(GL_LIGHTING)
    else:
        glDisable(GL_LIGHTING)

    for dv in data_visual:
        lid = dv[1]
        if link_filter is not None:
            if lid not in link_filter:
                continue
        if lid == -1:
            # Base link
            p, Q = pb_client.getBasePositionAndOrientation(model)
        else:
            # Non-base link
            link_state = pb_client.getLinkState(model, lid)
            p, Q = link_state[4], link_state[5]
        
        # World transform of the link
        T_joint = conversions.Qp2T(np.array(Q), np.array(p))
        
        glPushMatrix()
        gl_render.glTransform(T_joint)
        
        # If give, it will override the default value
        if isinstance(color, (np.ndarray, list)):
            link_color = color
        elif isinstance(color, dict):
            if lid in color:
                link_color = color[lid]
            elif 'default' in color:
                link_color = color['default']

        if draw_link: 
            render_geom(
                data_visual=dv,
                color=link_color)
        if draw_link_info: 
            render_geom_info(
                data_visual=dv,
                color=link_info_color, 
                scale=link_info_scale, 
                line_width=link_info_line_width,
                num_slice=link_info_num_slice)
        glPopMatrix()

def render_joints(pb_client, model):
    for j in range(pb_client.getNumJoints(model)):
        joint_info = pb_client.getJointInfo(model, j)
        joint_local_p, joint_local_Q, link_idx = joint_info[14], joint_info[15], joint_info[16]
        T_joint_local = conversions.Qp2T(
            np.array(joint_local_Q), np.array(joint_local_p))
        if link_idx == -1:
            link_world_p, link_world_Q = pb_client.getBasePositionAndOrientation(model)
        else:
            link_info = pb_client.getLinkState(model, link_idx)
            link_world_p, link_world_Q = link_info[0], link_info[1]
        T_link_world = conversions.Qp2T(
            np.array(link_world_Q), np.array(link_world_p))
        T_joint_world = np.dot(T_link_world, T_joint_local)
        # Render joint position
        glPushMatrix()
        gl_render.glTransform(T_joint_world)
        gl_render.render_point(np.zeros(3), radius=0.02, color=[0, 0, 0, 1])
        # Render joint axis depending on joint types
        joint_type = joint_info[2]
        if joint_type == pb_client.JOINT_FIXED:
            pass
        elif joint_type == pb_client.JOINT_REVOLUTE:
            axis = joint_info[13]
            gl_render.render_line(np.zeros(3), axis, color=[1, 0, 0, 1], line_width=1.0)
        elif joint_type == pb_client.JOINT_SPHERICAL:
            gl_render.render_transform(constants.eye_T(), scale=0.2)
        else:
            raise NotImplementedError()
        glPopMatrix()

def render_joint_geoms(
    pb_client, 
    model, 
    radius=0.025, 
    color=[0.5, 0.5, 0.5, 1],
    link_info_scale=1.0,
    link_info_color=[0, 0, 0, 1],
    link_info_line_width=1.0,
    link_info_num_slice=32,
    ):
    for j in range(pb_client.getNumJoints(model)):
        joint_info = pb_client.getJointInfo(model, j)
        joint_local_p, joint_local_Q, link_idx = joint_info[14], joint_info[15], joint_info[16]
        T_joint_local = conversions.Qp2T(
            np.array(joint_local_Q), np.array(joint_local_p))
        if link_idx == -1:
            link_world_p, link_world_Q = pb_client.getBasePositionAndOrientation(model)
        else:
            link_info = pb_client.getLinkState(model, link_idx)
            link_world_p, link_world_Q = link_info[0], link_info[1]
        T_link_world = conversions.Qp2T(
            np.array(link_world_Q), np.array(link_world_p))
        T_joint_world = np.dot(T_link_world, T_joint_local)
        # Render joint position
        glPushMatrix()
        gl_render.glTransform(T_joint_world)
        render_geom(
            data_visual=[-1, -1, pybullet.GEOM_SPHERE, [radius]],
            T=constants.EYE_T,
            color=color)
        render_geom_info(
            data_visual=[-1, -1, pybullet.GEOM_SPHERE, [radius]],
            T=constants.EYE_T,
            color=link_info_color, 
            scale=link_info_scale, 
            line_width=link_info_line_width,
            num_slice=link_info_num_slice)
        glPopMatrix()

def render_model(
    pb_client, 
    model, 
    draw_link=True, 
    draw_link_info=True, 
    draw_joint=False, 
    draw_joint_geom=True, 
    ee_indices=None, 
    color=None,
    link_info_scale=1.0,
    link_info_color=[0, 0, 0, 1],
    link_info_line_width=1.0,
    link_info_num_slice=32,
    lighting=True,
    ):
    if draw_link or draw_link_info:
        render_links(
            pb_client=pb_client,
            model=model, 
            color=color, 
            draw_link=draw_link,
            draw_link_info=draw_link_info,
            link_info_scale=link_info_scale,
            link_info_color=link_info_color,
            link_info_line_width=link_info_line_width,
            link_info_num_slice=link_info_num_slice,
            lighting=lighting)
    if draw_joint_geom:
        render_joint_geoms(pb_client, model)
    glDisable(GL_DEPTH_TEST)
    if draw_joint:
        render_joints(pb_client, model)
        render_links_pos(pb_client, model)
    glEnable(GL_DEPTH_TEST)


def render_contacts(pb_client, model, scale_h=0.0005, scale_r=0.01, color=[1.0,0.1,0.0,0.5]):
    data = pb_client.getContactPoints(model)
    for d in data:
        p, n, l = np.array(d[6]), np.array(d[7]), d[9]
        print(d[10], d[11], d[12], d[13])
        p1 = p
        p2 = p + n * l * scale_h
        gl_render.render_arrow(p1, p2, D=scale_r, color=color)