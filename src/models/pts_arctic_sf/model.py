import torch
import torch.nn as nn

import common.ld_utils as ld_utils

from .sub_model import Downsampler, RegressHead
from common.xdict import xdict
from src.nets.backbone.utils import get_backbone_info
from src.nets.hand_heads.hand_hmr import HandHMR
from src.nets.hand_heads.mano_head import MANOHead
from src.nets.obj_heads.obj_head import ArtiHead
from src.nets.obj_heads.obj_hmr import ObjectHMR


from common.xdict import xdict
from src.nets.backbone.utils import get_backbone_info
from src.nets.obj_heads.obj_head import ArtiHead
from src.nets.pointnet import PointNetfeat


class PtsArcticSF(nn.Module):
    def __init__(self, backbone, focal_length, img_res, args):
        super(PtsArcticSF, self).__init__()
        self.args = args
        if backbone == "resnet50":
            from src.nets.backbone.resnet import resnet50 as resnet
        elif backbone == "resnet18":
            from src.nets.backbone.resnet import resnet18 as resnet
        else:
            assert False
        self.backbone = resnet(pretrained=True)
        feat_dim = get_backbone_info(backbone)["n_output_channels"]
        self.arti_head = ArtiHead(focal_length=focal_length, img_res=img_res)

        img_down_dim = 512
        img_mid_dim = 512
        pt_out_dim = 512
        self.down = nn.Sequential(
            nn.Linear(feat_dim, img_mid_dim),
            nn.ReLU(),
            nn.Linear(img_mid_dim, img_down_dim),
            nn.ReLU(),
        )  # downsize image features

        pt_shallow_dim = 512
        pt_mid_dim = 512
        self.point_backbone = PointNetfeat(
            input_dim=3 + img_down_dim,
            shallow_dim=pt_shallow_dim,
            mid_dim=pt_mid_dim,
            out_dim=pt_out_dim,
        )
        pts_dim = pt_shallow_dim + pt_out_dim
        self.dist_head_or = RegressHead(pts_dim)
        self.dist_head_ol = RegressHead(pts_dim)
        self.dist_head_ro = RegressHead(pts_dim)
        self.dist_head_lo = RegressHead(pts_dim)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.num_v_sub = 195  # mano subsampled
        self.num_v_o_sub = 300 * 2  # object subsampled
        self.num_v_sub_sub = 70
        self.num_v_o_sub_sub = 100*2  # object
        self.downsampling_r = Downsampler(self.num_v_sub, self.num_v_sub_sub)
        self.downsampling_l = Downsampler(self.num_v_sub, self.num_v_sub_sub)
        self.downsampling_o = Downsampler(self.num_v_o_sub, self.num_v_o_sub_sub)

        # Arctic_sf
        mano_feat_dim = (pts_dim + 1) * self.num_v_sub_sub
        oject_feat_dim = (pts_dim + 2) * self.num_v_o_sub_sub

        self.head_r = HandHMR(mano_feat_dim, is_rhand=True, n_iter=3)
        self.head_l = HandHMR(mano_feat_dim, is_rhand=False, n_iter=3)

        self.head_o = ObjectHMR(oject_feat_dim, n_iter=3)

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

    def get_features_from_field_part(self, inputs, meta_info):
        images = inputs["img"]
        img_num = images.shape[0]
        points_r = meta_info["v0.r"].permute(0, 2, 1)[:, :, 21:]
        points_l = meta_info["v0.l"].permute(0, 2, 1)[:, :, 21:]
        points_o = meta_info["v0.o"].permute(0, 2, 1)
        points_all = torch.cat((points_r, points_l, points_o), dim=2)

        features = self.backbone(images)
        feat_vec = features.view(features.shape[0], features.shape[1], -1).sum(dim=2)

        features = self.avgpool(features).view(features.shape[0], -1)
        features = self.down(features)

        self.num_mano_pts = points_r.shape[2]
        self.num_object_pts = points_o.shape[2]

        img_feat_all = features[:, :, None].repeat(
            1, 1, self.num_mano_pts * 2 + self.num_object_pts
        )

        pts_all_feat = torch.cat((points_all, img_feat_all), dim=1)
        pts_all_feat = self.point_backbone(pts_all_feat)[0] # (:, 1024, 990)
        
        pts_r_feat, pts_l_feat, pts_o_feat = torch.split(
            pts_all_feat,
            [self.num_mano_pts, self.num_mano_pts, self.num_object_pts],
            dim=2,
        )

        dist_ro = self.dist_head_ro(pts_r_feat) # (:, 195)
        dist_lo = self.dist_head_lo(pts_l_feat) # (:, 195)
        dist_or = self.dist_head_or(pts_o_feat) # (:, 600)
        dist_ol = self.dist_head_ol(pts_o_feat) # (:, 600)

        features_r = torch.cat([pts_r_feat, dist_ro.unsqueeze(1)], dim=1)
        features_l = torch.cat([pts_l_feat, dist_lo.unsqueeze(1)], dim=1)
        features_o = torch.cat([pts_o_feat, dist_or.unsqueeze(1), dist_ol.unsqueeze(1)], dim=1)

        features_r = self.downsampling_r(features_r).view(img_num, -1)
        features_l = self.downsampling_l(features_l).view(img_num, -1)
        features_o = self.downsampling_o(features_o).view(img_num, -1)

        return feat_vec, features_r, features_l, features_o

    def forward(self, inputs, meta_info):
        feat_vec, features_r, features_l, features_o = self.get_features_from_field_part(inputs, meta_info)
        
        hmr_output_r = self.head_r.forward(features_r, use_pool=False)
        hmr_output_l = self.head_l.forward(features_l, use_pool=False)
        hmr_output_obj = self.head_o.forward(features_o, use_pool=False)

        query_names = meta_info["query_names"]
        K = meta_info["intrinsics"]
        
        mano_output_r = self.mano_r(
            rotmat=hmr_output_r["pose"],
            shape=hmr_output_r["shape"],
            K=K,
            cam=hmr_output_r["cam_t.wp"]
        )

        mano_output_l = self.mano_l(
            rotmat=hmr_output_l["pose"],
            shape=hmr_output_l["shape"],
            K=K,
            cam=hmr_output_l["cam_t.wp"]
        )

        # fwd mesh when in val or vis
        arti_output = self.arti_head(
            rot=hmr_output_obj["rot"],
            angle=hmr_output_obj["radian"],
            query_names=query_names,
            cam=hmr_output_obj["cam_t.wp"],
            K=K,
        )

        mano_output_r["cam_t.wp.init.r"] = hmr_output_r["cam_t.wp.init"]
        mano_output_l["cam_t.wp.init.l"] = hmr_output_l["cam_t.wp.init"]
        arti_output["cam_t.wp.init"] = hmr_output_obj["cam_t.wp.init"]

        mano_output_r = ld_utils.prefix_dict(mano_output_r, "mano.")
        mano_output_l = ld_utils.prefix_dict(mano_output_l, "mano.")
        arti_output = ld_utils.prefix_dict(arti_output, "object.")
        output = xdict()
        output.merge(mano_output_r)
        output.merge(mano_output_l)
        output.merge(arti_output)
        output["feat_vec"] = feat_vec.cpu().detach() # use field feat
        return output
