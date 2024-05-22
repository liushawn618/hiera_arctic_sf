import os
import json

class colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'

exp_folder = "logs/3558f1342"

seq_count = {}
eval_path = os.path.join(exp_folder, "eval")

for seq in os.listdir(eval_path):
    seq_count[seq] = len(os.listdir(os.path.join(eval_path, seq, "images")))

required = {"gt_mesh", "gt_mesh_l", "gt_mesh_r", "gt_mesh_obj"}
render_path = os.path.join(exp_folder, "render")

seq_missing = {}
seq_all_missing = set(seq_count.keys()) - set(os.listdir(render_path))

for seq in seq_all_missing:
        print(f"{colors.RED}Missing {seq}{colors.RESET}")
        seq_missing[seq] = "all"

for seq in seq_count:
    if seq in seq_all_missing:
        continue
    required_copy = required.copy()
    for render_mode in os.listdir(os.path.join(render_path, seq)):
        if render_mode not in required:
            continue
        else:
            required_copy.remove(render_mode)
            if render_mode.split("_")[-1] in ["l", "r", "obj"]:
                suffix = "mask"
            else:
                suffix = "rgb"
            try:
                actual_count = len(os.listdir(os.path.join(render_path, seq, render_mode, "images", suffix)))
            except FileNotFoundError:
                seq_missing.setdefault(seq, {})
                seq_missing[seq][render_mode] = "all"
                print(f"{colors.RED}Missing {render_mode} in {seq}{colors.RESET} : missing image")
            if actual_count != seq_count[seq]:
                seq_missing.setdefault(seq, {})
                seq_missing[seq][render_mode] = f"{actual_count}/{seq_count[seq]}"
                print(f"{colors.RED}Missing {actual_count}/{seq_count[seq]} {render_mode} in {seq}{colors.RESET}")
    for all_missing in required_copy:
        seq_missing.setdefault(seq, {})
        seq_missing[seq][all_missing] = "all"
        print(f"{colors.RED}Missing {all_missing} in {seq}{colors.RESET}")

with open("missing.json", "w") as f:
    json.dump(seq_missing, f, indent=4)
