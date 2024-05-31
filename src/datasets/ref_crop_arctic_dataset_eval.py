import cv2
import os
import torch
import json
import os.path as op

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset

import src.datasets.dataset_utils as dataset_utils
from common.data_utils import read_img
import common.data_utils as data_utils
from .ref_crop_arctic_dataset import RefCropArcticDataset
from src.datasets.dataset_utils import get_valid, pad_jts2d

class RefCropArcticDatasetEval(RefCropArcticDataset):

    def getitem(self, imgname, load_rgb=True):
        inputs, targets, meta_info = super().getitem_eval(imgname, load_rgb=True)
        inputs = self._load_ref_img(imgname, inputs)
        return inputs, targets, meta_info