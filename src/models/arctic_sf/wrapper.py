from src.callbacks.loss.loss_arctic_sf import compute_loss
from src.callbacks.process.process_arctic import process_data
from src.callbacks.vis.visualize_arctic import visualize_all
from src.models.arctic_sf.model import ArcticSF
from src.models.generic.wrapper import GenericWrapper
from src.models.config import ModelConfig

from functools import wraps
def mask_loss_deco(fn):
    from src.callbacks.loss.loss_mask import MaskLoss
    loss = MaskLoss()

    @wraps(fn)
    def wrapper(pred, targets, meta_info, args):
        v = loss.construct_mesh_scenes_o(pred, meta_info)
        a = loss.get_mask(v)
        return fn(pred, targets, meta_info, args)
    return wrapper

class ArcticSFWrapper(GenericWrapper):
    def __init__(self, args):
        super().__init__(args)
        self.model = ArcticSF(
            # backbone="resnet50",
            backbone=ModelConfig.backbone,
            focal_length=args.focal_length,
            img_res=args.img_res,
            args=args,
        )
        self.process_fn = process_data
        self.loss_fn = compute_loss
        self.metric_dict = [
            "cdev",
            "mrrpe",
            "mpjpe.ra",
            "aae",
            "success_rate",
        ]

        self.vis_fns = [visualize_all]

        self.num_vis_train = 1
        self.num_vis_val = 1

    def inference(self, inputs, meta_info):
        return super().inference_pose(inputs, meta_info)
