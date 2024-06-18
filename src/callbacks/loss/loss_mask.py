import torch
import numpy as np
import trimesh
import os.path as op

from aitviewer.headless import HeadlessRenderer
from aitviewer.renderables.billboard import Billboard
from aitviewer.scene.camera import OpenCVCamera
from aitviewer.renderables.meshes import Meshes
from aitviewer.utils.so3 import aa2rot_numpy

import common.viewer as viewer_utils
from common.body_models import build_layers, seal_mano_mesh
from common.xdict import xdict
from src.extraction.interface import prepare_data
from src.extraction.keys.vis_pose import KEYS as keys
from common.viewer import SEGM_IDS, materials
from common.object_tensors import OBJECTS

class MaskLoss:
    def __init__(self, loss_mode=None):
        self.loss_mode = ["o", "r", "l"] if loss_mode is None else loss_mode
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.v = HeadlessRenderer()
        self.layers = build_layers(device)
        self.obj_faces = {
            obj_name:
                trimesh.load(
                    f"./data/arctic_data/data/meta/object_vtemplates/{obj_name}/mesh.obj",
                    process=False).faces 
            for obj_name in OBJECTS
            }

    def reset(self):
        self.v.reset()

    def normize_targets(self, targets):
        pass

    def __setup_viewer(self, v:HeadlessRenderer, cam_t):
        # setup billboard
        focal = 1000.0
        rows = 224
        cols = 224
        K = np.array([[focal, 0, rows / 2.0], [0, focal, cols / 2.0], [0, 0, 1]])
        Rt = np.zeros((1, 3, 4)) # 1frame, 1scene
        Rt[:, :, 3] = cam_t
        Rt[:, :3, :3] = np.eye(3)
        Rt[:, 1:3, :3] *= -1.0
        camera = OpenCVCamera(K, Rt, cols, rows, viewer=v)
        v.scene.add(camera)
        v.scene.camera.load_cam()
        v.set_temp_camera(camera)

        fps = 30
        v.run_animations = True  # autoplay
        v.run_animations = False  # autoplay
        v.playback_fps = fps
        v.scene.fps = fps
        v.scene.origin.enabled = False
        v.scene.floor.enabled = False
        v.auto_set_floor = False
        v.scene.floor.position[1] = -3
        return v

    def construct_mesh_scenes_o(self, inputs, meta_info):
        v = self.v
        
        obj_names = meta_info["query_names"]
        cam_t = inputs["object.cam_t"]
        v3d_o = inputs["object.v.cam"]
        v3d_o -= cam_t[:, None, :]
        f3d_o_list = [
            self.obj_faces[obj_name] for obj_name in obj_names
        ]
        
        self.__setup_viewer(v, cam_t)

        v3d = v3d_o
        meshes = []
        rotation_flip = aa2rot_numpy(np.array([1, 0, 0]) * np.pi)
        for v3d, f3d in zip(torch.split(v3d_o, 1, dim=0), f3d_o_list):
            # WARN: auto diff may disabled
            v3d = v3d.numpy()
            mesh = Meshes(
                    v3d,
                    f3d,
                    vertex_colors=None,
                    name="object",
                    flat_shading=False,
                    draw_edges=False,
                    material=materials["light-blue"],
                    rotation=rotation_flip,
                )
            meshes.append(mesh)
            v.scene.add(mesh)
        return v
        
    def get_mask(self, v:HeadlessRenderer):
        v._init_scene()
        nodes_uid = {node.name: node.uid for node in v.scene.collect_nodes()}
        my_cmap = {
            uid: [SEGM_IDS[name], SEGM_IDS[name], SEGM_IDS[name]]
            for name, uid in nodes_uid.items()
            if name in SEGM_IDS.keys()
        }
        return np.array(v.get_mask(color_map=my_cmap)).astype(np.uint8)

    def get_depth(v):
        return np.array(v.get_depth()).astype(np.float16)
