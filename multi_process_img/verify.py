import os

from multi_process_img.config import verify_path, SEQS_DIR

missing = {}

img_dir = ["gt_mesh/images/rgb",
"gt_mesh_l/images/mask",
"gt_mesh_r/images/mask",
"gt_mesh_obj/images/mask"]

check_dir = ["../crop_mask", "../crop_image"]

for seq in os.listdir(SEQS_DIR):
    seq_dir = os.path.join(SEQS_DIR, seq)