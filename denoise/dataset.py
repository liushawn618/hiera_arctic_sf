import numpy as np
import torch
import os
import bisect
import logging
import time

from .lib.utils.geometry_utils import *
from .lib.base_dataset import BaseDataset

logger = logging.getLogger(__name__)

H36M_IMG_SHAPE=1000

class ArcticOutputDataset(BaseDataset):

    def __init__(self, window_size, stride=1, seqs_dir="logs/3558f1342/eval", phase='train'):
        """
        logs/3558f1342/eval
        """

        if phase == 'train':
            self.phase = phase  # 'train' | 'test'
        elif phase == 'test':
            self.phase = phase
        else:
            raise NotImplementedError(
                "Unknown phase type! Valid phase: [\'train\',\'test\']. You can edit the code for additional implements"
            )

        self.slide_window_size = window_size
        self.evaluate_slide_window_step = stride

        self.seqs_dir = seqs_dir

        ground_truth_data={}
        predicted_data={}
        seq_frame_num = {}

        start_time = time.time()

        for seq in os.listdir(seqs_dir):
            pred_dir = os.path.join(seqs_dir, seq, "preds")
            target_dir = os.path.join(seqs_dir, seq, "targets")

            num_frames = None
            
            for pred_file in os.listdir(pred_dir):
                if not pred_file.startswith("pred."):
                    continue
                key = ".".join(pred_file.split(".")[1:-1])
                pred_file_path = os.path.join(pred_dir, pred_file)
                val = torch.load(pred_file_path)
                if num_frames is None:
                    num_frames = val.shape[0]
                    if num_frames < self.slide_window_size:
                        logger.warning(f"desperating to load too short seq {seq}")
                        break
                predicted_data.setdefault(seq,{})
                predicted_data[seq][key] = val
                assert num_frames == predicted_data[seq][key].shape[0]

            if num_frames < self.slide_window_size:
                continue

            for target_file in os.listdir(target_dir):
                if not target_file.startswith("targets."):
                    continue
                key = ".".join(target_file.split(".")[1:-1])
                target_file_path = os.path.join(target_dir, target_file)
                ground_truth_data.setdefault(seq,{})
                ground_truth_data[seq][key] = torch.load(target_file_path)
                assert num_frames == ground_truth_data[seq][key].shape[0]

            seq_frame_num[seq] = num_frames

        self.start_idx:dict[int, str] = {}
        _cur_start = 0
        self.seq_data_len = {seq:max(frame_num - self.slide_window_size + 1, 0) for seq,frame_num in seq_frame_num.items()}
        for seq, data_len in self.seq_data_len.items():
            self.start_idx[_cur_start] = seq
            _cur_start += data_len

        self.total_frame_num = sum(seq_frame_num.values())
        self.total_data_num = _cur_start
        self.sequence_num = len(seq_frame_num.keys())
        
        self.ground_truth_data = ground_truth_data
        self.predicted_pose = predicted_data
        self.seq_frame_num = seq_frame_num

        end_time = time.time()
        logger.info(f"Loaded {self.total_data_num} data pieces in {end_time-start_time:.2f}s")

    def __len__(self):
        # if self.phase == "train":
            return self.total_frame_num
        # elif self.phase == "test":
        #     return self.sequence_num

    def __getitem__(self, index):
        if self.phase == "train":
            return self.get_data(index)

        elif self.phase == "test":
            return self.get_test_data(index)

    def get_data(self, index):
        keys = sorted(self.start_idx.keys())
        seq_start_idx = bisect.bisect(keys, index)-1
        seq_start_idx = keys[seq_start_idx]

        seq = self.start_idx[seq_start_idx]

        frame_num = self.seq_frame_num[seq]

        gt_data = self.ground_truth_data[seq]
        pred_data = self.predicted_pose[seq]

        def padding(x, padding_size):
            return torch.concatenate(
                (x, torch.zeros((padding_size, ) + tuple(x.shape[1:]))),
                dim=0)

        ret_gt_data = {}
        ret_pred_data = {}
        if self.slide_window_size <= frame_num:
            start_idx = index - seq_start_idx
            assert start_idx < self.seq_data_len[seq]
            end_idx = start_idx + self.slide_window_size
            
            for key in gt_data.keys():
                ret_gt_data[key] = gt_data[key][start_idx:end_idx] #.reshape((self.slide_window_size, -1))
            for key in pred_data.keys():
                ret_pred_data[key] = pred_data[key][start_idx:end_idx]
        else:
            assert False
        return {"gt": ret_gt_data, "pred": ret_pred_data}

    # TODO
    def get_test_data(self, index):
        ground_truth_data_len = len(self.ground_truth_data_imgname[index])
        detected_data_len = len(self.detected_data_imgname[index])

        if ground_truth_data_len != detected_data_len:
            raise ImportError(
                "Detected data is not the same size with ground_truth data!")

        if self.return_type == '3D':
            gt_data = self.ground_truth_data_joints_3d[index]
            pred_data = self.detected_data_joints_3d[index]
        elif self.return_type == '2D':
            gt_data = (self.ground_truth_data_joints_2d[index]-H36M_IMG_SHAPE/2)/(H36M_IMG_SHAPE/2) # normalization
            pred_data = (self.detected_data_joints_2d[index]-H36M_IMG_SHAPE/2)/(H36M_IMG_SHAPE/2) # normalization
        
        elif self.return_type == 'smpl':
            gt_data = self.ground_truth_data_pose[index].reshape(
                ground_truth_data_len, -1)
            pred_data = self.detected_data_pose[index].reshape(
                ground_truth_data_len, -1)

            gt_shape = self.ground_truth_data_shape[index].reshape(
                ground_truth_data_len, -1)
            pred_shape = self.detected_data_shape[index].reshape(
                ground_truth_data_len, -1)
            gt_data = np.concatenate((gt_data, gt_shape), axis=-1)
            pred_data = np.concatenate((pred_data, pred_shape), axis=-1)

        if self.slide_window_size <= ground_truth_data_len:
            start_idx=np.arange(0,ground_truth_data_len-self.slide_window_size+1,self.evaluate_slide_window_step)
            gt_data_=[]
            pred_data_=[]
            for idx in start_idx:
                gt_data_.append(gt_data[idx:idx+self.slide_window_size,:])
                pred_data_.append(pred_data[idx:idx+self.slide_window_size,:])

            gt_data=np.array(gt_data_)
            pred_data=np.array(pred_data_)
        else:
            gt_data = np.concatenate((
                gt_data,
                np.zeros(
                    tuple((self.slide_window_size - ground_truth_data_len, )) +
                    tuple(gt_data.shape[1:]))),
                                     axis=0)[np.newaxis, :]
            pred_data = np.concatenate((
                pred_data,
                np.zeros(
                    tuple((self.slide_window_size - ground_truth_data_len, )) +
                    tuple(pred_data.shape[1:]))),
                                       axis=0)[np.newaxis, :]
        return {"gt": gt_data, "pred": pred_data}
        
