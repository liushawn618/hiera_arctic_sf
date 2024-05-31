import torch

import src.callbacks.process.process_arctic as process_arctic

def prepare_mano_template(batch_size, mano_layer, mesh_sampler, is_right, pose=None, beta=None):
    root_idx = 0

    # Generate T-pose template mesh
    if pose is None:
        template_pose = torch.zeros((1, 48)).cuda()
    else:
        template_pose = pose
    if beta is None:
        template_betas = torch.zeros((1, 10)).cuda()
    else:
        template_betas = beta
    out = mano_layer(
        betas=template_betas,
        hand_pose=template_pose[:, 3:],
        global_orient=template_pose[:, :3],
        transl=None,
    )
    template_3d_joints = out.joints
    template_vertices = out.vertices
    template_vertices_sub = mesh_sampler.downsample(template_vertices, is_right)

    # normalize
    template_root = template_3d_joints[:, root_idx, :]
    template_3d_joints = template_3d_joints - template_root[:, None, :]
    template_vertices = template_vertices - template_root[:, None, :]
    template_vertices_sub = template_vertices_sub - template_root[:, None, :]

    # concatinate template joints and template vertices, and then duplicate to batch size
    ref_vertices = torch.cat([template_3d_joints, template_vertices_sub], dim=1)
    ref_vertices = ref_vertices.expand(batch_size, -1, -1)

    ref_vertices_full = torch.cat([template_3d_joints, template_vertices], dim=1)
    ref_vertices_full = ref_vertices_full.expand(batch_size, -1, -1)
    return ref_vertices, ref_vertices_full

def prepare_object_template(batch_size, object_tensors, query_names, angles=None, rot=None):
    if angles is None:
        template_angles = torch.zeros((batch_size, 1)).cuda()
    else:
        template_angles = angles
    if rot is None:
        template_rot = torch.zeros((batch_size, 3)).cuda()
    else:
        template_rot = rot
    out = object_tensors.forward(
        angles=template_angles,
        global_orient=template_rot,
        transl=None,
        query_names=query_names,
    )
    ref_vertices = out["v_sub"]
    parts_idx = out["parts_ids"]

    mask = out["mask"]

    ref_mean = ref_vertices.mean(dim=1)[:, None, :]
    ref_vertices -= ref_mean

    v_template = out["v"]
    return (ref_vertices, parts_idx, v_template, mask)

def prepare_templates(
    batch_size,
    mano_r,
    mano_l,
    mesh_sampler,
    arti_head,
    query_names,
    reference_pred:dict
):
    ref_pred = reference_pred
    v0_r, v0_r_full = prepare_mano_template(
        batch_size, mano_r, mesh_sampler, is_right=True, pose=ref_pred["pose_r"], beta=ref_pred["beta_r"]
    )
    v0_l, v0_l_full = prepare_mano_template(
        batch_size, mano_l, mesh_sampler, is_right=False, pose=ref_pred["pose_l"], beta=ref_pred["beta_l"]
    )
    (v0_o, pidx, v0_full, mask) = prepare_object_template(
        batch_size,
        arti_head.object_tensors,
        query_names,
        rot=ref_pred["rot"],
        angles=ref_pred["angles"]
    )
    CAM_R, CAM_L, CAM_O = list(range(100))[-3:]
    cams = (
        torch.FloatTensor([CAM_R, CAM_L, CAM_O]).view(1, 3, 1).repeat(batch_size, 1, 3)
        / 100
    )
    cams = cams.to(v0_r.device)
    return (
        v0_r,
        v0_l,
        v0_o,
        pidx,
        v0_r_full,
        v0_l_full,
        v0_full,
        mask,
        cams,
    )

def process_data(models, inputs, targets, meta_info, mode, args): # mode no use
    batch_size = meta_info["intrinsics"].shape[0]

    (
        v0_r,
        v0_l,
        v0_o,
        pidx,
        v0_r_full,
        v0_l_full,
        v0_o_full,
        mask,
        cams,
    ) = prepare_templates(
        batch_size,
        models["mano_r"],
        models["mano_l"],
        models["mesh_sampler"],
        models["arti_head"],
        meta_info["query_names"],
        meta_info["ref_pred"]
    )

    meta_info["v0.r"] = v0_r
    meta_info["v0.l"] = v0_l
    meta_info["v0.o"] = v0_o
    meta_info["cams0"] = cams
    meta_info["parts_idx"] = pidx
    meta_info["v0.r.full"] = v0_r_full
    meta_info["v0.l.full"] = v0_l_full
    meta_info["v0.o.full"] = v0_o_full
    meta_info["mask"] = mask

    inputs, targets, meta_info = process_arctic.process_data(
        models, inputs, targets, meta_info, mode, args, field_max=args.max_dist
    )

    return inputs, targets, meta_info
