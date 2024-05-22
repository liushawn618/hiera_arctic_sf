import os
import json 
import multiprocessing as mp
import subprocess
import random
import argparse
import sys

from easydict import EasyDict

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num_workers", type=int,
                    default=1, help="number of workers")

parser.add_argument("--exp_folder", type=str, default="logs/3558f1342")
parser.add_argument("--task_file", type=str, default="task.txt")
parser.add_argument("--shuffle", action="store_true")  # 随机打乱seq

parser.add_argument("-q", "--quiet", dest="is_quiet", action="store_true")
args = parser.parse_args()

class Config:
    python = "/home/lx/anaconda3/envs/arctic_env/bin/python"
    gpus = [1, 2]

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


def split_tasks(tasks: list[str], num_parts: int, shuffle=False):
    tasks = tasks.copy()
    if shuffle:
        random.shuffle(tasks)
    tasks_per_parts = len(tasks) // num_parts
    remain_task_num = len(tasks) % num_parts
    tasks_split = []
    for i in range(num_parts):
        if i < remain_task_num:
            tasks_split.append(tasks[i * tasks_per_parts + i:(i + 1) * tasks_per_parts + i + 1])
        else:
            tasks_split.append(tasks[i * tasks_per_parts + remain_task_num:(i + 1) * tasks_per_parts + remain_task_num])
    return tasks_split

error_log = {}

def run_task(runner_id:int, task:str, is_quiet=False, envs=None):
    if envs is None:
        envs = os.environ.copy()
    mode_cmd = {
        "gt_mesh_l": ["--render_type=mask", "--mode=gt_mesh_l"],
        "gt_mesh_r": ["--render_type=mask", "--mode=gt_mesh_r"],
        "gt_mesh_obj": ["--render_type=mask", "--mode=gt_mesh_obj"],
        "gt_mesh": ["--render_type=rgb", "--mode=gt_mesh", "--no_model"]
    }
    seq_name, mode = task.split(".")
    base_cmd = [Config.python, "scripts_method/visualizer.py",
                    f"--exp_folder={args.exp_folder}", f"--seq_name={seq_name}", "--headless"]
    cmd = base_cmd + mode_cmd[mode]
    actual_cmd = " ".join(cmd)
    try:
        if is_quiet:
            result = subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=envs)
        else:
            print(f"{runner_id}.{mode}>> {actual_cmd}")
            result = subprocess.run(cmd, env=envs)
        # os.system(actual_cmd)
    except Exception as e:
        print(
            f"{runner_id}.{mode}>> {seq_name} {colors.RED}failed{colors.RESET}")
        error_log[task] = f"mainprocess error>> {e}"
        
    else:
        if result.returncode != 0:
            print(
                f"{runner_id}.{mode}>> {seq_name} {colors.RED}failed{colors.RESET}")
            error_log[task] = f"subprocess error>> {result.stderr.decode()}"
            print(result.stderr.decode())
        else:
            print(
                f"{runner_id}.{mode}>> {seq_name} {colors.GREEN}succeeded{colors.RESET}")

def run_tasks(id: int, tasks: list, args, envs=None):
    for task in tasks:
        run_task(id, task, args.is_quiet, envs)
    print(f"{id}>> {colors.YELLOW}finished{colors.RESET}")

if __name__ == "__main__":
    num_workers = args.num_workers
    with open(args.task_file, "r") as f:
        task_list = f.read().splitlines()
    splited_tasks = split_tasks(task_list, num_workers, args.shuffle)
    processes = []
    
    try:
        for i in range(num_workers):
            envs = os.environ.copy()
            envs["CUDA_VISIBLE_DEVICES"] = str(Config.gpus[i % len(Config.gpus)])
            p = mp.Process(target=run_tasks, args=(
                i, splited_tasks[i], args, envs))
            p.start()
            processes.append(p)

        for process in processes:
            process.join()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        with open("error.log.json", "w") as f:
            json.dump(error_log, f)
        for process in processes:
            process.terminate()
    with open("error.log.json", "w") as f:
        json.dump(error_log, f)
       