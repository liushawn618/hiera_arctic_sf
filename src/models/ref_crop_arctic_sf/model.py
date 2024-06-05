import torch
import torch.nn as nn

import common.ld_utils as ld_utils
from common.xdict import xdict
from src.nets.backbone.utils import get_backbone_info
from src.nets.hand_heads.hand_hmr import HandHMR
from src.nets.hand_heads.mano_head import MANOHead
from src.nets.obj_heads.obj_head import ArtiHead
from src.nets.obj_heads.obj_hmr import ObjectHMR

from src.models.config import ModelConfig

class RefCropArcticSF(nn.Module):
    def __init__(self, backbone, focal_length, img_res, args):
        super(RefCropArcticSF, self).__init__()
        self.args = args
        self.no_crop = args.no_crop
        self.backbone = ModelConfig.get_backbone(backbone)
        feat_dim = get_backbone_info(backbone)["n_output_channels"]
        self.head_r = HandHMR(feat_dim, is_rhand=True, n_iter=3)
        self.head_l = HandHMR(feat_dim, is_rhand=False, n_iter=3)
        self.head_o = ObjectHMR(feat_dim, n_iter=3)

        self.head_r_cam = HandHMR(feat_dim, is_rhand=True, n_iter=3)
        self.head_l_cam = HandHMR(feat_dim, is_rhand=False, n_iter=3)
        self.head_o_cam = ObjectHMR(feat_dim, n_iter=3)

        self.mano_r = MANOHead(
            is_rhand=True, focal_length=focal_length, img_res=img_res
        )

        self.mano_l = MANOHead(
            is_rhand=False, focal_length=focal_length, img_res=img_res
        )

        self.arti_head = ArtiHead(focal_length=focal_length, img_res=img_res)
        self.mode = "train"
        self.img_res = img_res
        self.focal_length = focal_length

    def forward(self, inputs, meta_info):
        if self.no_crop:
            images_r=inputs["img"]
            images_l=inputs["img"]
            images_obj=inputs["img"]
        else:
            images_r=inputs["ref_img_r_rgb"]
            images_l=inputs["ref_img_l_rgb"]
            images_obj=inputs["ref_img_o_rgb"]
        
        features_r = self.backbone(images_r)
        features_l = self.backbone(images_l)
        features_obj = self.backbone(images_obj)
        
        # feat_vec_r = features.view(features_r.shape[0], features_r.shape[1], -1).sum(dim=2)

        hmr_output_r_ref = self.head_r(features_r)
        hmr_output_l_ref = self.head_l(features_l)
        hmr_output_obj_ref = self.head_o(features_obj)


        images = inputs["img"]
        query_names = meta_info["query_names"]
        K = meta_info["intrinsics"]
        features = self.backbone(images)
        
        
        feat_vec = features.view(features.shape[0], features.shape[1], -1).sum(dim=2)

        hmr_output_r = self.head_r_cam(features)
        hmr_output_l = self.head_l_cam(features)
        hmr_output_o = self.head_o_cam(features)

        # weak perspective
        root_r = hmr_output_r["cam_t.wp"]
        root_l = hmr_output_l["cam_t.wp"]
        root_o = hmr_output_o["cam_t.wp"]

        mano_output_r = self.mano_r(
            rotmat=hmr_output_r_ref["pose"],
            shape=hmr_output_r_ref["shape"],
            K=K,
            cam=root_r
        )

        mano_output_l = self.mano_l(
            rotmat=hmr_output_l_ref["pose"],
            shape=hmr_output_l_ref["shape"],
            K=K,
            cam=root_l,
        )

        # fwd mesh when in val or vis
        arti_output = self.arti_head(
            rot=hmr_output_obj_ref["rot"],
            angle=hmr_output_obj_ref["radian"],
            query_names=query_names,
            cam=root_o,
            K=K,
        )

        mano_output_r["cam_t.wp.init.r"] = hmr_output_r["cam_t.wp.init"]
        mano_output_l["cam_t.wp.init.l"] = hmr_output_l["cam_t.wp.init"]
        arti_output["cam_t.wp.init"] = hmr_output_o["cam_t.wp.init"]

        mano_output_r = ld_utils.prefix_dict(mano_output_r, "mano.")
        mano_output_l = ld_utils.prefix_dict(mano_output_l, "mano.")
        arti_output = ld_utils.prefix_dict(arti_output, "object.")
        output = xdict()
        output.merge(mano_output_r)
        output.merge(mano_output_l)
        output.merge(arti_output)
        output["feat_vec"] = feat_vec.cpu().detach()
        return output
