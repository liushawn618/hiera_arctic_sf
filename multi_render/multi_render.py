import os
import multiprocessing as mp
# mp.set_start_method('spawn')
import subprocess
import random
import argparse
import json

from easydict import EasyDict

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num_workers", type=int,
                    default=1, help="number of workers")

parser.add_argument("--exp_folder", type=str, default="logs/3558f1342")
parser.add_argument("--num_parts", type=str, default=None)
parser.add_argument("--choose", type=str, default=None)
parser.add_argument("--filter_file", type=str, default=None)
parser.add_argument("--shuffle", action="store_true")  # 随机打乱seq

parser.add_argument("-q", "--quiet", dest="is_quiet", action="store_true")

# 无参全跑；有continue只跑没跑的；再加rerun额外跑失败的
parser.add_argument("--rerun", dest="is_rerun", action="store_true")  # 失败重跑
parser.add_argument("--continue", dest="is_continue",
                    action="store_true")  # 继续


def parse_args():
    config = parser.parse_args()
    args = EasyDict(vars(config))
    if args.num_parts is None:
        args.num_parts = args.num_workers
    return args

class Config:
    python = "/home/lx/anaconda3/envs/arctic_env/bin/python"
    render_log_dir = "render_log"
    gpus = [0, 1, 2]


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


def get_filter(path: str):
    if path is None:
        return None
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]

def get_seqs(seq_folder, filtered_seqs=None, record=None, is_continue=False, is_rerun=False):
    def filter(dir_path):
        result = True
        folder_name = os.path.basename(dir_path)
        if filtered_seqs is not None:
            if folder_name in filtered_seqs:
                return False
        # if record is not None:
        #     if folder_name in record.get("succeeded",[]):
        #         result = result and not is_continue
        #     if folder_name in record.get("failed",[]):
        #         result = result and (is_rerun or not is_continue)
        return result and os.path.isdir(os.path.join(dir_path, 'meta_info'))

    # 获取指定目录下的所有直接子文件夹
    subfolders = [os.path.join(seq_folder, f) for f in os.listdir(
        seq_folder) if os.path.isdir(os.path.join(seq_folder, f))]
    seqs = []
    # 遍历每个子文件夹
    for folder in subfolders:
        # 检查文件夹下是否存在 meta_info 子文件夹
        if filter(folder):
            folder_name = os.path.basename(folder)
            seqs.append(folder_name)
    return seqs


def split_seqs(seqs: list[str], num_parts: int, shuffle=False):
    seqs = seqs.copy()
    if shuffle:
        random.shuffle(seqs)
    seqs_per_worker = len(seqs) // num_parts
    remain_seq_num = len(seqs) % num_parts
    seqs_split = []
    for i in range(num_parts):
        if i < remain_seq_num:
            seqs_split.append(seqs[i * seqs_per_worker + i:(i + 1) * seqs_per_worker + i + 1])
        else:
            seqs_split.append(seqs[i * seqs_per_worker + remain_seq_num:(i + 1) * seqs_per_worker + remain_seq_num])
    # seqs_split = [seqs[i * seqs_per_worker + i:(i + 1) * seqs_per_worker + i] if i < remain_seq_num
    #               else seqs[i * seqs_per_worker + remain_seq_num:(i + 1) * seqs_per_worker + remain_seq_num + 1]
    #               for i in range(num_parts)]
    return seqs_split


def redistri_seqs(seqs: list[list[str]], choose: str):
    num_parts = len(seqs)
    if choose is None:
        return seqs
    def parse_choose(choose: str):
        # choose example: 1,2,3-7
        result:list[int]=[]
        for index_range in choose.split(","):
            range_split = index_range.split("-")
            if len(range_split) == 1:
                result += [int(index_range)]
            elif len(range_split) == 2:
                result += list(range(int(range_split[0]), int(range_split[1])+1))
            else:
                raise ValueError(f"Invalid choose: {choose}")
        return result
    filter_index = parse_choose(choose)
    result:list[str] = []
    for index in filter_index:
        result += seqs[index]
    return split_seqs(result, num_parts)

def render_seqs(id: int, seqs: list, args, envs=None):
    if envs is None:
        envs = os.environ.copy()
    log_dir = os.path.join(args.exp_folder, Config.render_log_dir)
    log_file = f"{id}.log"
    record = {
        "succeeded": [],
        "failed": []
    }

    tasks = {
        "l": ["--render_type=mask", "--mode=gt_mesh_l"],
        "r": ["--render_type=mask", "--mode=gt_mesh_r"],
        "obj": ["--render_type=mask", "--mode=gt_mesh_obj"],
        "rgb": ["--render_type=rgb", "--mode=gt_mesh", "--no_model"]
    }

    for seq_name in seqs:
        base_cmd = [Config.python, "scripts_method/visualizer.py",
                    f"--exp_folder={args.exp_folder}", f"--seq_name={seq_name}", "--headless"]
        for task_name, task_cmd in tasks.items():
            cmd = base_cmd + task_cmd
            actual_cmd = " ".join(cmd)
            try:
                if args.is_quiet:
                    result = subprocess.run(
                        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=envs)
                else:
                    print(f"{i}.{task_name}>> {actual_cmd}")
                    result = subprocess.run(cmd, env=envs)
                # os.system(actual_cmd)
            except Exception as e:
                record["failed"].append(f"{seq_name}.{task_name}")
                print(
                    f"{i}.{task_name}>> {seq_name} {colors.RED}failed{colors.RESET}")
            else:
                record["succeeded"].append(f"{seq_name}.{task_name}")
                print(
                    f"{i}.{task_name}>> {seq_name} {colors.GREEN}succeeded{colors.RESET}")

    with open(os.path.join(log_dir, log_file), "w") as f:
        json.dump(record, f, indent=4)

    print(f"{id}>> {colors.YELLOW}finished{colors.RESET}")


def collect_logs(log_dir: str):
    result = {
        "succeeded": [],
        "failed": []
    }
    for log_file in os.listdir(log_dir):
        if not log_file.endswith(".log"):
            continue
        with open(os.path.join(log_dir, log_file), "r") as f:
            single_result = json.load(f)
        result["succeeded"].extend(single_result["succeeded"])
        result["failed"].extend(single_result["failed"])
    return result


if __name__ == "__main__":
    args = parse_args()
    num_workers = args.num_workers
    log_dir = os.path.join(args.exp_folder, Config.render_log_dir)
    log_file = os.path.join(log_dir, "all.log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    record = None
    if args.is_continue:
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                try:
                    record = json.load(f)
                except:
                    record = None

    seq_folder = os.path.join(args.exp_folder, "eval")
    splited_seqs = split_seqs(get_seqs(seq_folder=seq_folder, filtered_seqs=get_filter(args.filter_file), record=record,
                              is_continue=args.is_continue, is_rerun=args.is_rerun), args.num_parts, shuffle=args.shuffle)
    splited_seqs = redistri_seqs(splited_seqs, args.choose)
    processes: list[mp.Process] = []
    for i in range(num_workers):
        envs = os.environ.copy()
        envs["CUDA_VISIBLE_DEVICES"] = str(Config.gpus[i % len(Config.gpus)])
        p = mp.Process(target=render_seqs, args=(
            i, splited_seqs[i], args, envs))
        p.start()
        processes.append(p)

    for process in processes:
        process.join()

    with open(os.path.join(log_dir, "all.log"), "w") as f:
        json.dump(collect_logs(log_dir), f, indent=4)
