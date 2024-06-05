import torch
from typing import Literal
from loguru import logger

class ModelConfig:
    backbone:Literal["resnet18", "resnet50", "hiera"] = "resnet50"
    
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
            # model = torch.hub.load("facebookresearch/hiera", model="hiera_base_224", pretrained=True, checkpoint="mae_in1k_ft_in1k")
            model = torch.hub.load("pretrained", model="hiera_base_224", pretrained=True, source="local")
            logger.info("loaded hiera backbone successfully")
            return model
        else:
            raise NotImplementedError