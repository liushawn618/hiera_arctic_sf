import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--missing_json", type=str, default="missing.json")
parser.add_argument("--output_file", type=str, default="task.txt")
args = parser.parse_args()

subtasks = {"gt_mesh", "gt_mesh_l", "gt_mesh_r", "gt_mesh_obj"}

with open(args.missing_json, "r") as f:
    missing_json = json.load(f)

tasks = []

for seq, content in missing_json.items():
    if content == "all":
        tasks += [f"{seq}.{subtask}" for subtask in subtasks]
    else:
        for mode in content:
            tasks += [f"{seq}.{mode}"]

with open(args.output_file, "w") as f:
    for task in tasks:
        f.write(f"{task}\n")
