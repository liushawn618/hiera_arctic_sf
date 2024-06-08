import cv2
import os
import torch
import json
import os.path as op

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset

import src.datasets.dataset_utils as dataset_utils
from common.data_utils import read_img
import common.data_utils as data_utils
from .arctic_dataset import ArcticDataset
from src.datasets.dataset_utils import get_valid, pad_jts2d

class RefCropArcticDataset(ArcticDataset):
    target_size = (224, 224)

    def plain_process_rgb(self, rgb_img):
        args = self.args
        # augment parameters
        augm_dict = data_utils.augm_params(
            self.aug_data,
            args.flip_prob,
            args.noise_factor,
            args.rot_factor,
            args.scale_factor,
        )
        pn = augm_dict["pn"]
        rgb_img[:, :, 0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 0] * pn[0]))
        rgb_img[:, :, 1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 1] * pn[1]))
        rgb_img[:, :, 2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 2] * pn[2]))
        rgb_img = np.transpose(rgb_img.astype("float32"), (2, 0, 1)) / 255.0
        return rgb_img

    def _load_ref_img(self, imgname, inputs):
        # WARN modify dict inputs directly
        if not self.ref_online: # offline
            img_basename = os.path.basename(imgname).split(".")[0]
            ref_img_name = f"{img_basename}.png"
            seq_name = "_".join(imgname.split("/")[-4:-1])
            modes = ["gt_mesh_l", "gt_mesh_r", "gt_mesh_obj"]
            sub_types = ["crop_image"]#, "crop_mask"]
            for mode, mode_alias in zip(modes, ["l", "r", "o"]):
                for sub_type, sub_type_alias in zip(sub_types, ["rgb", "mask"]):
                    ref_img_path = os.path.join(self.reference_exp_dir, seq_name, mode, "images", sub_type, ref_img_name)
                    ref_img, img_status = read_img(ref_img_path, self.target_size + (3,), nonexsit_ok=True)
                    ref_img = cv2.resize(ref_img, self.target_size)
                    ref_img = self.plain_process_rgb(ref_img)
                    ref_img = torch.from_numpy(ref_img).float()
                    inputs[f"ref_img_{mode_alias}_{sub_type_alias}"] = ref_img # ref_img_l_rgb
        return inputs

    def getitem(self, imgname, load_rgb=True):
        inputs, targets, meta_info = super().getitem(imgname, load_rgb=True)
        inputs = self._load_ref_img(imgname, inputs)
        return inputs, targets, meta_info
    def __init__(self, args, split, seq=None):
        super().__init__(args, split, seq=seq)
        self.reference_exp_dir = args.ref_crop_folder # logs/3558f1342
        if args.ref_mode == "online":
            self.ref_online = True
        else:
            self.ref_online = False