import numpy as np
import os
from hiera_diffusion_policy.env.nonprehensile.rsuite.path import *
import open3d as o3d
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import array_to_string, find_elements, xml_path_completion


class TestObject(MujocoXMLObject):
    """
    Coke can object (used in PickPlace)
    """

    def __init__(self, name:str, joints=True, obj_type='all'):
        _name = name.replace('_target', '') if 'target' in name else name
        xml_path = os.path.join(rs_assets_path(), 'objects/{}/{}.xml'.format(_name, _name))
        stl_file = xml_path.replace('.xml', '.stl')
        mesh = o3d.io.read_triangle_mesh(stl_file)
        box = mesh.get_axis_aligned_bounding_box()
        self.size = box.max_bound
        super().__init__(
            xml_path,
            name=name,
            joints="default" if joints else None,
            obj_type=obj_type,
            duplicate_collision_geoms=True,
        )
        
    
    @property
    def bottom_offset(self):
        return np.array([0, 0, -1 * self.size[2]])

    @property
    def top_offset(self):
        return np.array([0, 0, self.size[2]])
        
    @property
    def horizontal_radius(self):
        return np.linalg.norm(self.size[0:2], 2)



class TestYCBObject(MujocoXMLObject):
    """
    Milk carton object (used in PickPlace)
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/testycb.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class TestYCBVisualObject(MujocoXMLObject):
    """
    Visual fiducial of cereal box (used in PickPlace)

    Fiducial objects are not involved in collision physics.
    They provide a point of reference to indicate a position.
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/testycb-visual.xml"),
            name=name,
            joints=None,
            obj_type="visual",
            duplicate_collision_geoms=True,
        )


class TriangularPrismObject(MujocoXMLObject):

    def __init__(self, name):
        xml_path = os.path.join(rs_assets_path(), 'objects/triangular_prism/triangular_prism_7cm.xml')
        super().__init__(
            xml_path,
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )
        self.stl_path = os.path.join(rs_assets_path(), 'objects/triangular_prism/triangular_prism_7cm.stl')


class TBoxObject(MujocoXMLObject):

    def __init__(self, name, visual=False):
        xml_path = os.path.join(rs_assets_path(), 'objects/tbox/tbox.xml')
        super().__init__(
            xml_path,
            name=name,
            joints=None if visual else [dict(type="free", damping="0.0005")],
            obj_type='visual' if visual else "all",
            duplicate_collision_geoms=True,
        )
        self.stl_path = os.path.join(rs_assets_path(), 'objects/tbox/tbox.stl')