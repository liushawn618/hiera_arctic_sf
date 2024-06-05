import os
from easydict import EasyDict
def init_ref_args(args:EasyDict):
    if not args.method.startswith("ref_"):
        return args
    if args.ref_mode == "online":
        if args.ref_setup is None or args.ref_method is None:
            raise ValueError(f"invalid arg for --ref_setup={args.ref_setup}, --ref_method={args.ref_method}")
        if args.ref_setup != args.setup:
            # raise NotImplementedError(f"unsupported --ref_setup={args.ref_setup} != --setup={args.setup}, may cause unexpected error")
            print(f"WARN {args.ref_setup} != {args.setup}")
        if args.ref_ckpt is None:
            args.ref_ckpt = os.path.join(args.reference_exp_folder, "checkpoints/last.ckpt")
    return args