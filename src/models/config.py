import torch
from typing import Literal
from loguru import logger
from functools import wraps

def hiera_deco(func):

    @wraps(func)
    def extra_process(*args, **kwargs):
        kwargs.setdefault("return_intermediates", True)
        _, img_feat = func(*args, **kwargs)
        last_stage_img_feat = img_feat[-1]
        features = last_stage_img_feat.permute(0, 3, 1, 2)
        return features
    return extra_process

class ModelConfig:
    backbone:Literal["resnet18", "resnet50", "hiera"] = "resnet50" # effective when getbackbone(None)
    
    @staticmethod
    def get_backbone(backbone:str|None=None):
        if backbone is None:
            backbone = ModelConfig.backbone
        if backbone == "resnet50":
            from src.nets.backbone.resnet import resnet50 as resnet
            logger.info("using resnet50 backbone")
            return resnet(pretrained=True)
        elif backbone == "resnet18":
            from src.nets.backbone.resnet import resnet18 as resnet
            logger.info("using resnet18 backbone")
            return resnet(pretrained=True)
        elif backbone == "hiera":
            logger.info("using hiera backbone")
            model = torch.hub.load("pretrained", model="hiera_base_224", pretrained=True, source="local")
            model.forward = hiera_deco(model.forward)
            logger.info("loaded hiera backbone successfully")
            return model
        else:
            raise NotImplementedError