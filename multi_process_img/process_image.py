import cv2
import os
import glob
import argparse
import tqdm
import logging
import time
import shutil
import numpy as np

boxType = tuple[int, int, int, int]
logger = logging.getLogger(__name__)

# python multi_process_img/process_image.py --seq_dir logs/3558f1342/render/s06_microwave_grab_01_3 --save_dir temp_output --save_mode direct --work_mode mask
parser = argparse.ArgumentParser()
parser.add_argument("--seq_dir", type=str, default=None,
                    help="eg: logs/3558f1342/render/s06_microwave_grab_01_3")
parser.add_argument("--save_dir", type=str, default=None,
                    help="eg: logs/3558f1342/render/s06_microwave_grab_01_3 or {save_parent_dir}/s06_microwave_grab_01_3")
parser.add_argument("--save_mode", type=str, choices=["no", "ln", "direct", "clear"], default="direct",
                    help="no effect on storage; use ln to store in mapping dir; directily store ; clear to delete the raw image")
parser.add_argument("--map_dir", type=str, default=None,
                    help="eg: /data_mapping/arctic")
parser.add_argument("--work_mode", type=str, choices=["crop", "mask"], default="crop")
parser.add_argument("--out_size", type=int, default=None, help="output image size; eg: 1000")
parser.add_argument("--perform_eval", action="store_true")

args = parser.parse_args()
seq_dir = args.seq_dir
seq_name = [seq_name for seq_name in reversed(
    seq_dir.split("/")) if "_" in seq_name][0]

if args.save_dir is None:
    save_dir = seq_dir
else:
    save_dir = args.save_dir

target_size = None

raw_path = os.path.join(seq_dir, "gt_mesh/images/rgb")
l_path = os.path.join(seq_dir, "gt_mesh_l/images/mask")
r_path = os.path.join(seq_dir, "gt_mesh_r/images/mask")
o_path = os.path.join(seq_dir, "gt_mesh_obj/images/mask")

l_save_path = os.path.join(save_dir, "gt_mesh_l/images/crop_image")
r_save_path = os.path.join(save_dir, "gt_mesh_r/images/crop_image")
o_save_path = os.path.join(save_dir, "gt_mesh_obj/images/crop_image")


def prepare_dir():
    if args.save_mode == "no":
        return
    for out_dir in [l_save_path, r_save_path, o_save_path]:
        if os.path.exists(out_dir):
            if os.path.islink(out_dir):
                source_path = os.readlink(out_dir)
                shutil.rmtree(source_path)
                tqdm.tqdm.write(f"removed ln_src {source_path}")
                os.unlink(out_dir)
                tqdm.tqdm.write(f"removed ln {out_dir}")
            elif os.path.isdir(out_dir):
                shutil.rmtree(out_dir)
                tqdm.tqdm.write(f"removed dir {out_dir}")
            else:
                raise NotImplementedError
        if args.save_mode == "direct":
            os.makedirs(out_dir, exist_ok=True)
        if args.save_mode == "ln":
            src_path = os.path.join(
                args.map_dir, out_dir)
            os.makedirs(src_path, exist_ok=True)
            out_dir_parent = os.path.dirname(out_dir)
            os.makedirs(out_dir_parent, exist_ok=True)
            os.symlink(src_path, out_dir)


def overlap_boxes(box1: boxType, box2: boxType):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # 计算大矩形框的坐标和大小
    x_min = min(x1, x2)
    y_min = min(y1, y2)
    x_max = max(x1 + w1, x2 + w2)
    y_max = max(y1 + h1, y2 + h2)

    new_box = (x_min, y_min, x_max - x_min, y_max - y_min)

    # 判断是否重叠
    if (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1):
        return False, new_box
    return True, new_box


def get_border(image, expand_distance=40):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 阈值操作将图像二值化
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours_len = len(contours)
    if contours_len == 0:
        return False, None

    boxes: list[boxType] = []

    for contour in contours:

        # 获取原始边界框的坐标和尺寸
        x, y, w, h = cv2.boundingRect(contour)

        # 计算原始边界框的中心坐标
        center_x = x + w // 2
        center_y = y + h // 2

        # 计算边界框的边长
        edge = max(w, h) + 2 * expand_distance

        # 调整边界框为正方形，并以原始边界框的中心为中心
        x_new = max(0, center_x - edge // 2)
        y_new = max(0, center_y - edge // 2)

        # 确保不超出图像范围
        x_new = min(x_new, image.shape[1] - edge)
        y_new = min(y_new, image.shape[0] - edge)

        box = (x_new, y_new, edge, edge)
        boxes.append(box)
    normal_flag = True
    if len(boxes) == 1:
        return normal_flag, boxes[0]
    # if contours_len > 2:
    #     logger.warning(f"detected {contours_len} boxes")
    result = boxes[0]
    for overlapping_box in boxes[1:]:
        is_overlapping, result = overlap_boxes(result, overlapping_box)
        normal_flag = normal_flag and is_overlapping
    return normal_flag, result


def crop(image, border, target_size=target_size):
    if border is None:
        return None
    x, y, w, h = border
    result = image[y:y+h, x:x+w]
    if target_size is not None:
        result = cv2.resize(result, target_size)
    return result

def mask(image, border, target_size=target_size):
    if border is None:
        return None
    x, y, w, h = border
    mask = np.zeros_like(image)
    mask[y:y+h, x:x+w] = (255, 255, 255)
    result = cv2.bitwise_and(image, mask)
    if target_size is not None:
        result = cv2.resize(result, target_size)
    return result

def save_image(file_path, image):
    if image is None:
        return

    def create_dir(dir: str):
        if not os.path.exists(dir):
            os.makedirs(dir)

    # 获取文件夹路径
    folder_path = os.path.dirname(file_path)
    create_dir(folder_path)

    # 保存图像
    cv2.imwrite(file_path, image)


if __name__ == "__main__":
    print(f"{seq_name} save mode is {args.save_mode}")
    prepare_dir()
    if args.save_mode == "clear":
        exit()
    png_list = [os.path.basename(png_path) for png_path in glob.glob(
        os.path.join(raw_path, "*.png"))]

    out_size = args.out_size
    if out_size is not None:
        target_size = (out_size, out_size)  # (1000 ,1000)
    elif out_size == -1:
        target_size = cv2.imread(os.path.join(raw_path, png_list[0])).shape[:2]

    if args.work_mode == "crop":
        work_fn = crop
    elif args.work_mode == "mask":
        work_fn = mask
    else:
        raise NotImplementedError

    if args.perform_eval:
        total_border_time = 0
        total_crop_time = 0
        total_time = 0

    for png_file_name in tqdm.tqdm(png_list, desc=f"{seq_name}"):
        raw_image = cv2.imread(os.path.join(raw_path, png_file_name))

        l_image = cv2.imread(os.path.join(l_path, png_file_name))
        r_image = cv2.imread(os.path.join(r_path, png_file_name))
        o_image = cv2.imread(os.path.join(o_path, png_file_name))

        if args.perform_eval:
            border_start = time.time()

        flag_l, l_border = get_border(l_image)
        flag_r, r_border = get_border(r_image)
        flag_o, o_border = get_border(o_image)

        if args.perform_eval:
            border_end = time.time()
            border_interval = border_end - border_start

        if not (flag_o and flag_l and flag_r):
            unormal_names = [name for name, flag, border in zip(["l", "r", "o"], [flag_l, flag_r, flag_o], [
                                                                l_border, r_border, o_border]) if not flag and border is not None]
            if len(unormal_names) > 0:
                tqdm.tqdm.write(
                    f"{seq_name}.{png_file_name} seems unormal in {unormal_names} with multi boxes detected while no overlapping")

        if args.perform_eval:
            work_start = time.time()

            work_fn(raw_image, l_border)
            work_fn(raw_image, r_border)
            work_fn(raw_image, o_border)
            work_fn(l_image, l_border)
            work_fn(r_image, r_border)
            work_fn(o_image, o_border)

            work_end = time.time()
            crop_interval = work_end - work_start
            total = border_interval + crop_interval

            total_border_time += border_interval
            total_crop_time += crop_interval
            total_time += total

            continue

        if args.save_mode == "no":
            continue

        save_image(os.path.join(l_save_path, png_file_name), work_fn(raw_image, l_border))
        save_image(os.path.join(r_save_path, png_file_name), work_fn(raw_image, r_border))
        save_image(os.path.join(o_save_path, png_file_name), work_fn(raw_image, o_border))

        # save_image(os.path.join(l_path, "../crop_image",
        #            png_path), crop(raw_image, l_border))
        # save_image(os.path.join(r_path, "../crop_image",
        #            png_path), crop(raw_image, r_border))
        # save_image(os.path.join(o_path, "../crop_image",
        #            png_path), crop(raw_image, o_border))

        # save_image(os.path.join(l_path, "../crop_mask",
        #            png_path), crop(l_image, l_border))
        # save_image(os.path.join(r_path, "../crop_mask",
        #            png_path), crop(r_image, r_border))
        # save_image(os.path.join(o_path, "../crop_mask",
        #            png_path), crop(o_image, o_border))

    if args.perform_eval:
        print(f"{seq_name} total border time({total_border_time/total_time*100:.2f}%): {total_border_time}, total crop time({total_crop_time/total_time*100:.2f}%): {total_crop_time}")
    print(f"{seq_name} done")