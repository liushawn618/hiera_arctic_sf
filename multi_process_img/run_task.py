import os
import argparse
import random
import subprocess
import multiprocessing as mp
import shutil
from io import TextIOWrapper

from multi_process_img.config import task_path, SEQS_DIR, PYTHON, colors, MAPPING_DIR

# python multi_process_img/run_task.py -n 36 -q --work_mode=mask --seqs_dir=logs/3558f1342/render --save_dir=logs/3558f1342/masked_render --ln_rate=0.4 --shuffle

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num_workers", type=int,
                    default=1, help="number of workers")
parser.add_argument("--seqs_dir", type=str,
                    default=SEQS_DIR, help=f"example: {SEQS_DIR}")
parser.add_argument("--save_dir", type=str,
                   help="default: {seqs_dir}/render")
parser.add_argument("--work_mode", type=str, default="crop")
parser.add_argument("--ln_rate", type=float, default=0.4)
parser.add_argument("--shuffle", action="store_true")  # 随机打乱seq

parser.add_argument("-q", "--quiet", dest="is_quiet", action="store_true")
parser.add_argument("-c", "--clear", dest="is_clear", action="store_true")

args = parser.parse_args()

if args.save_dir is None:
    args.save_dir = os.path.join(args.seqs_dir, "render")

SEQS_DIR = args.seqs_dir

def get_log(idx:int):
    log_dir = os.path.join("multi_process_img", "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    return open(os.path.join(log_dir, f"{idx}.log"), "w")

def get_seqs():
    if not os.path.exists(task_path):
        return os.listdir(SEQS_DIR)
    return []
    
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

def run_task(runner_id:int, seq:str, is_quiet=False, envs=None, log_file:int|TextIOWrapper=subprocess.DEVNULL, is_ln=False, is_clear=False):
    if envs is None:
        envs = os.environ.copy()
    seq_dir = os.path.join(SEQS_DIR, seq)
    save_dir = os.path.join(args.save_dir, seq)
    cmd = [PYTHON, "multi_process_img/process_image.py",
                    f"--seq_dir={seq_dir}", f"--save_dir={save_dir}", f"--map_dir={MAPPING_DIR}", f"--work_mode={args.work_mode}"]
    save_mode = "ln" if is_ln else "direct"
    if is_clear:
        save_mode = "clear"
    cmd += [f"--save_mode={save_mode}"]
    actual_cmd = " ".join(cmd)
    try:
        if is_quiet:
            result = subprocess.run(
                cmd, stdout=log_file, stderr=subprocess.PIPE, env=envs)
        else:
            print(f"{colors.CYAN}{runner_id}{colors.RESET}>> {actual_cmd}")
            result = subprocess.run(cmd, env=envs)
        # os.system(actual_cmd)
    except Exception as e:
        print(
            f"{colors.CYAN}{runner_id}{colors.RESET}>> {seq} {colors.RED}failed{colors.RESET}")
        
    else:
        if result.returncode != 0:
            print(
                f"{colors.CYAN}{runner_id}{colors.RESET}>> {seq} {colors.RED}failed{colors.RESET}")
            print(result.stderr.decode())
        else:
            print(
                f"{colors.CYAN}{runner_id}{colors.RESET}>> {seq} {colors.GREEN}succeeded{colors.RESET}")
            if type(log_file) is TextIOWrapper:
                log_file.write(f"{seq} finished\n")

def run_tasks(id: int, tasks: list, args, envs=None, ln_flag:list[bool]=None, is_clear=False):
    log_file = get_log(id)
    for i, task in enumerate(tasks):
        if ln_flag is None:
            run_task(id, task, args.is_quiet, envs, log_file=log_file, is_clear=is_clear)
        else:
            run_task(id, task, args.is_quiet, envs, log_file=log_file, is_ln=ln_flag[i], is_clear=is_clear)
    print(f"{colors.CYAN}{id}{colors.RESET}>> {colors.YELLOW}finished{colors.RESET}")
    log_file.close()

def clear_log():
    print("clear log...")
    log_dir = "multi_process_img/logs"
    shutil.rmtree(log_dir, ignore_errors=True)
    os.makedirs(log_dir, exist_ok=True)
    print("clear log finished")

if __name__ == "__main__":
    clear_log()
    num_workers = args.num_workers
    ln_rate = args.ln_rate

    task_list = get_seqs()
    task_num = len(task_list)
    print(f"total tasks: {task_num}")
    
    ln_nums = int(task_num * ln_rate)
    is_ln = [True]*ln_nums + [False]*(task_num-ln_nums)

    splited_tasks = split_tasks(task_list, num_workers, args.shuffle)
    splited_ln_flag = split_tasks(is_ln, num_workers, shuffle=True)
    processes = []
    
    for i in range(num_workers):
        envs = os.environ.copy()
        p = mp.Process(target=run_tasks, args=(
            i, splited_tasks[i], args, envs, splited_ln_flag[i], args.is_clear))
        p.start()
        processes.append(p)

    for process in processes:
        process.join()
