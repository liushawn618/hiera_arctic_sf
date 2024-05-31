import os
import sys

TASK_PATH_IN_MODULE = ".temp/task.txt"
VERIFY_PATH_IN_MODULE = ".temp/missing.json"
SEQS_DIR = "logs/3558f1342/render"
MAPPING_DIR = "/data_mapping/arctic"
PYTHON = "/home/lx/anaconda3/envs/arctic_env/bin/python"

module_dir = os.path.dirname(sys.argv[1])

task_path = os.path.join(module_dir, TASK_PATH_IN_MODULE)
verify_path = os.path.join(module_dir, VERIFY_PATH_IN_MODULE)

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