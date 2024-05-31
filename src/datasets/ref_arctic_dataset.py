import os
import torch

from pytorch3d.transforms.rotation_conversions import matrix_to_axis_angle

from .arctic_dataset import ArcticDataset

# TODO collect to cuda -> thing2dev
class RefPredLoader:
    def __init__(self, pred_output_dir):
        self.pose_l = torch.load(os.path.join(pred_output_dir, "pred.mano.pose.l.pt")).cuda()
        self.pose_r = torch.load(os.path.join(pred_output_dir, "pred.mano.pose.r.pt")).cuda()

        self.pose_l = matrix_to_axis_angle(self.pose_l.reshape(-1, 3, 3)).reshape(-1, 48).cpu()
        self.pose_r = matrix_to_axis_angle(self.pose_r.reshape(-1, 3, 3)).reshape(-1, 48).cpu()

        self.beta_l = torch.load(os.path.join(pred_output_dir, "pred.mano.beta.l.pt")).cpu()
        self.beta_r = torch.load(os.path.join(pred_output_dir, "pred.mano.beta.r.pt")).cpu()

        self.angles = torch.load(os.path.join(pred_output_dir, "pred.object.radian.pt")).unsqueeze(1).cpu()
        self.rot = torch.load(os.path.join(pred_output_dir, "pred.object.rot.pt")).cpu()

        self.len = self.pose_l.shape[0]

    def __getitem__(self, index):
        if index >= self.len:
            return {
            "pose_l": torch.zeros((1, 48)),
            "pose_r": torch.zeros((1, 48)),
            "beta_l": self.beta_l[[index],:],
            "beta_r": self.beta_r[[index],:],
            "angles": self.angles[[index],:],
            "rot": self.rot[[index],:]
        }

        return {
            "pose_l": self.pose_l[[index],:],
            "pose_r": self.pose_r[[index],:],
            "beta_l": self.beta_l[[index],:],
            "beta_r": self.beta_r[[index],:],
            "angles": self.angles[[index],:],
            "rot": self.rot[[index],:]
        }
    
class RefPredDataset:
    sub_folder = "eval"
    def __init__(self, reference_exp_dir:str, lazy=True): # logs/3558f1342
        self.reference_exp_dir = reference_exp_dir
        self.seqs2loader = {}
        if lazy:
            return
        # WARN: eval may mismatch to cropped_images
        seqs_dir = os.path.join(reference_exp_dir, self.sub_folder)
        for seq_name in os.listdir(seqs_dir):
            pred_output_dir = os.path.join(seqs_dir, seq_name, "preds")
            self.seqs2loader[seq_name] = RefPredLoader(pred_output_dir=pred_output_dir)

    def get_pred(self, seq_name:str, idx:int):
        loader = self.seqs2loader.get(seq_name)
        if loader is None:
            pred_output_dir = os.path.join(self.reference_exp_dir, "eval", seq_name, "preds")
            loader = self.seqs2loader[seq_name] = RefPredLoader(pred_output_dir=pred_output_dir)
        return loader[idx]

class RefArcticDataset(ArcticDataset):
    def getitem(self, imgname, load_rgb=True): # ./arctic_data/data/images/s02/microwave_use_02/6/00227.jpg
        inputs, targets, meta_info = super().getitem(imgname, load_rgb=True)
        if self.ref_online:
            return inputs, targets, meta_info
        raise NotImplementedError("offline ref mode not implemented")
        seq_name = "_".join(imgname.split("/")[-4:-1])
        img_idx = int(imgname.split("/")[-1].split(".")[0])
        meta_info["ref_pred"] = self.ref_dataset.get_pred(seq_name=seq_name, idx=img_idx)
        return inputs, targets, meta_info
    
    def __init__(self, args, split, seq=None):
        super().__init__(args, split, seq=seq)
        if args.ref_mode == "online":
            self.ref_online = True
            reference_exp_dir = args.reference_exp_folder # logs/3558f1342
            self.ref_dataset = RefPredDataset(reference_exp_dir, lazy=False)
        else:
            self.ref_online = False
            self.ref_dataset = None
        
