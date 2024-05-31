import argparse

def add_ref_parser(parser:argparse.ArgumentParser):
    # train field with ref mode
    parser.add_argument("--ref_exp_folder", "--reference_exp_folder", type=str, dest="reference_exp_folder", default="logs/3558f1342")
    parser.add_argument("--ref_mode", type=str, choices=["online", "offline"], default="online")
    parser.add_argument("--ref_setup", type=str, choices=["p1", "p2"], default=None)
    parser.add_argument("--ref_method", type=str, default=None)
    parser.add_argument("--ref_ckpt", type=str, default=None)
    return parser
