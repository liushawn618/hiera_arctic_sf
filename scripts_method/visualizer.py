import argparse
import sys

from easydict import EasyDict

sys.path = ["."] + sys.path
import os.path as op
from glob import glob

import numpy as np
from loguru import logger

from common.viewer import ARCTICViewer, ViewerData
from common.xdict import xdict
from aitviewer.configuration import CONFIG as C

import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_folder", type=str, default="logs/3558f1342")
    parser.add_argument("--angle", type=float, default=None)
    parser.add_argument("--zoom_out", type=float, default=0.5)
    parser.add_argument("--seq_name", type=str, default="s05_box_grab_01_1")
    parser.add_argument("--render_type", type=str, default="video")
    parser.add_argument(
        "--mode",
        type=str,
        default="gt_mesh_l",
        choices=[
            "gt_mesh",
            "gt_mesh_l",
            "gt_mesh_r",
            "gt_mesh_obj",
            "pred_mesh",
            "gt_field_r",
            "gt_field_l",
            "pred_field_r",
            "pred_field_l",
        ],
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no_image",action="store_true")
    parser.add_argument("--no_model",action="store_true")

    config = parser.parse_args()
    args = EasyDict(vars(config))
    return args


class MethodViewer(ARCTICViewer):
    def load_data(self, exp_folder, seq_name, mode, no_image=False, no_model=False):
        logger.info("Creating meshes")

        # check if we are loading gt or pred
        if "pred_mesh" in mode or "pred_field" in mode:
            flag = "pred"
        elif "gt_mesh" in mode or "gt_field" in mode:
            flag = "targets"
        else:
            assert False, f"Unknown mode {mode}"

        exp_key = exp_folder.split("/")[1]
        images_path = op.join(exp_folder, "eval", seq_name, "images")
        # images_path = "/home/lx/reproduce/hand/arctic/data/arctic_data/data/cropped_images/s01/box_grab_01/4"

        # load mesh
        meshes_all = xdict()

        print(f"Specs: {exp_key} {seq_name} {flag}")
        if "_mesh" in mode:
            from src.mesh_loaders.pose import construct_meshes

            meshes, data = construct_meshes(
                exp_folder, seq_name, flag, None, zoom_out=None
            )
            meshes_all.merge(meshes)
            if "_r" in mode:
                meshes_all.pop("left", None)
                meshes_all.pop("object", None)
            if "_l" in mode:
                meshes_all.pop("right", None)
                meshes_all.pop("object", None)
            if "_obj" in mode:
                meshes_all.pop("right", None)
                meshes_all.pop("left", None)
        elif "_field" in mode:
            from src.mesh_loaders.field import construct_meshes

            meshes, data = construct_meshes(
                exp_folder, seq_name, flag, mode, None, zoom_out=None
            )
            meshes_all.merge(meshes)
            if "_r" in mode:
                meshes_all.pop("left", None)
            if "_l" in mode:
                meshes_all.pop("right", None)
        else:
            assert False, f"Unknown mode {mode}"

        imgnames = sorted(glob(images_path + "/*"))
        num_frames = min(len(imgnames), data[f"{flag}.object.cam_t"].shape[0])

        # setup camera
        focal = 1000.0
        rows = 224
        cols = 224
        K = np.array([[focal, 0, rows / 2.0], [0, focal, cols / 2.0], [0, 0, 1]])
        
        cam_t = data[f"{flag}.object.cam_t"]
        cam_t = cam_t[:num_frames]
        Rt = np.zeros((num_frames, 3, 4))
        Rt[:, :, 3] = cam_t
        Rt[:, :3, :3] = np.eye(3)
        Rt[:, 1:3, :3] *= -1.0

        # pack data
        if no_image:
            imgnames = None
        
        data = ViewerData(Rt=Rt, K=K, cols=cols, rows=rows, imgnames=imgnames)
    
        batch = meshes_all, data
        self.check_format(batch)

        if no_model:
            meshes_all = xdict()

        logger.info("Done")
        return batch


def main(args):
    exp_folder = args.exp_folder
    seq_name = args.seq_name
    mode = args.mode
    render_type = args.render_type.split(",")
    C.window_width, C.window_height = 1000,1000
    viewer = MethodViewer(
        interactive=not args.headless,
        size=(2048, 2048),
        render_types=render_type,
    )
    
    logger.info(f"Rendering {seq_name} {mode}")
    batch = viewer.load_data(exp_folder, seq_name, mode, no_image=args.no_image)#, no_model=args.no_model)
    viewer.render_seq(batch, out_folder=op.join(exp_folder, "render", seq_name, mode), no_model=args.no_model)


if __name__ == "__main__":
    args = parse_args()
    main(args)
