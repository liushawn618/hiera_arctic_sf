import common.comet_utils as comet_utils
from common.ref_arg_init import init_ref_args
from src.parsers.parser import construct_args
import sys

args = construct_args()
experiment, args = comet_utils.init_experiment(args)
comet_utils.save_args(args, save_keys=["comet_key"])
if not args.mute and args.num_gpus == 1:
    input(f"total epochs:{args.num_epoch}\nlog dir:{args.log_dir}")
args = init_ref_args(args)

def get_ref_args(): # to build online wrapper
    args = construct_args()
    experiment, args = comet_utils.init_experiment(args)
    args = init_ref_args(args)
    args.setup = args.ref_setup
    args.method = args.ref_method
    return args