import argparse

from easydict import EasyDict

from common.args_utils import set_default_params
from src.parsers.generic_parser import add_generic_args
from src.parsers.ref_parser import add_ref_parser
from src.parsers.pts_parser import add_pts_parser

from src.models.config import ModelConfig


def construct_args(str_list_args:list[str]|None=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear_log", action="store_true")
    parser.add_argument("--name", default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--num_gpus", default=1, type=int, help="number of gpus to use")
    parser.add_argument("--backbone", default="resnet50", choices=["resnet18", "resnet50", "hiera"])
    parser.add_argument(
        "--method",
        type=str,
        default="arctic_sf",
        choices=[None, "arctic_sf", "pts_arctic_sf", "arctic_tf", "arctic_lstm", "field_sf", "field_lstm", "ref_field_sf", "ref_crop_arctic_sf"],
    )

    parser.add_argument("--exp_key", type=str, default=None)
    parser.add_argument("--extraction_mode", type=str, default=None)
    parser.add_argument("--img_feat_version", type=str, default=None)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--local", action="store_true")
    parser = add_generic_args(parser)
    parser = add_ref_parser(parser)
    parser = add_pts_parser(parser)
    args = EasyDict(vars(parser.parse_args(str_list_args)))

    if args.backbone is not None:
        ModelConfig.backbone = args.backbone

    if args.method in ["arctic_sf", "pts_arctic_sf", "arctic_tf"]:
        import src.parsers.configs.arctic_sf as config
    elif args.method in ["arctic_tf"]:
        import src.parsers.configs.arctic_transformer as config
    elif args.method in ["ref_crop_arctic_sf"]:
        import src.parsers.configs.ref_crop_arctic_sf as config
    elif args.method in ["arctic_lstm"]:
        import src.parsers.configs.arctic_lstm as config
    elif args.method in ["field_sf"]:
        import src.parsers.configs.field_sf as config
    elif args.method in ["field_lstm"]:
        import src.parsers.configs.field_lstm as config
    elif args.method in ["ref_field_sf"]:
        import src.parsers.configs.field_sf as config
    else:
        assert False

    default_args = (
        config.DEFAULT_ARGS_EGO if args.setup in ["p2"] else config.DEFAULT_ARGS_ALLO
    )
    args = set_default_params(args, default_args)

    args.focal_length = 1000.0
    args.img_res = 224
    args.rot_factor = 30.0
    args.noise_factor = 0.4
    args.scale_factor = 0.25
    args.flip_prob = 0.0
    args.img_norm_mean = [0.485, 0.456, 0.406]
    args.img_norm_std = [0.229, 0.224, 0.225]
    args.pin_memory = True
    args.shuffle_train = True
    args.seed = 1
    args.grad_clip = 150.0
    args.use_gt_k = False  # use weak perspective camera or the actual intrinsics
    args.speedup = True  # load cropped images for faster training
    # args.speedup = False # uncomment this to load full images instead
    args.max_dist = 0.10  # distance range the model predicts on
    args.ego_image_scale = 0.3

    if args.method in ["field_sf", "field_lstm"]:
        args.project = "interfield"
    else:
        args.project = "arctic"
    args.interface_p = None

    if args.fast_dev_run:
        args.num_workers = 0
        args.batch_size = 8
        args.trainsplit = "minitrain"
        args.valsplit = "minival"
        args.log_every = 5
        args.window_size = 3

    return args
