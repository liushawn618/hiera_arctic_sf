import torch
from pytorch3d.transforms.rotation_conversions import matrix_to_axis_angle

from src.models.field_sf.wrapper import FieldSFWrapper
from common.thing import thing2dev

from src.factory import fetch_model
import src.callbacks.process.process_referenced_field as process_referenced_field
from src.utils.const import get_ref_args

class ReferencedFieldSFWrapper(FieldSFWrapper):
    def __init__(self, args):
        super().__init__(args)
        self.process_fn = process_referenced_field.process_data
        if args.ref_mode == "online":
            self.online = True
            self.ref_wrapper = fetch_model(get_ref_args()).to("cuda") # WARN no cpu
            self.ref_wrapper.model.arti_head.object_tensors.to("cuda")
            self.ref_wrapper.load_state_dict(torch.load(args.ref_ckpt)["state_dict"])
        else:
            self.online = False
            self.ref_wrapper = None

    def add_ref_pred_in_meta_info(self, inputs, targets, meta_info, mode):
        with torch.no_grad():
            extracted_output = self.ref_wrapper.forward(inputs, targets, meta_info, "extract")
            extracted_output = thing2dev(extracted_output, self.device)
            ref_pred = {
                "pose_l": extracted_output.get("pred.mano.pose.l"),
                "pose_r": extracted_output.get("pred.mano.pose.r"),
                "beta_l": extracted_output.get("pred.mano.beta.l"),
                "beta_r": extracted_output.get("pred.mano.beta.r"),
                "angles": extracted_output.get("pred.object.radian"),
                "rot": extracted_output.get("pred.object.rot")
            }
            ref_pred["pose_l"] = matrix_to_axis_angle(ref_pred["pose_l"].reshape(-1, 3, 3)).reshape(-1, 48)
            ref_pred["pose_r"] = matrix_to_axis_angle(ref_pred["pose_r"].reshape(-1, 3, 3)).reshape(-1, 48)
            meta_info["ref_pred"] = ref_pred
        return inputs, targets, meta_info, mode

    def forward(self, inputs, targets, meta_info, mode):
        if self.online:
            inputs, targets, meta_info, mode = self.add_ref_pred_in_meta_info(inputs, targets, meta_info, mode)
        return super().forward(inputs, targets, meta_info, mode)

