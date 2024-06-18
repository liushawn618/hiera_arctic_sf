import argparse

def add_pts_parser(parser:argparse.ArgumentParser):
    # train field with ref mode
    parser.add_argument("--pts_h_num", "--pts_hand_point_num", type=int, dest="pts_hand_point_num", default=None)
    parser.add_argument("--pts_o_num", "--pts_object_point_num", type=int, dest="pts_object_point_num", default=None)
    return parser
