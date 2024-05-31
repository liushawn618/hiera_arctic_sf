import cv2
import os
import glob
import argparse
import tqdm
import logging
import time
import shutil

boxType = tuple[int, int, int, int]
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--seq_dir", type=str, default=None, help="eg: logs/3558f1342/render/s06_microwave_grab_01_3/render")
parser.add_argument("--save_mode", type=str, choices=["no", "ln", "direct", "clear"], default="direct")
parser.add_argument("--map_dir", type=str, default=None, help="eg: /data_mapping/arctic")

args = parser.parse_args()
seq_dir = args.seq_dir
seq_name = [seq_name for seq_name in reversed(seq_dir.split("/")) if "_" in seq_name][0]

raw_path = os.path.join(seq_dir, "gt_mesh/images/rgb")
l_path = os.path.join(seq_dir, "gt_mesh_l/images/mask")
r_path = os.path.join(seq_dir, "gt_mesh_r/images/mask")
o_path = os.path.join(seq_dir, "gt_mesh_obj/images/mask")
target_size = (224, 224)#(1000 ,1000)

def prepare_dir():
    if args.save_mode == "no":
        return
    for path in [l_path, r_path, o_path]:
        parent_dir = os.path.dirname(path)
        for relative_out_path in ["crop_image","crop_mask"]:
            out_dir = os.path.join(parent_dir, relative_out_path)
            if os.path.exists(out_dir):
                if os.path.islink(out_dir):
                    source_path = os.readlink(out_dir)
                    shutil.rmtree(source_path)
                    tqdm.tqdm.write(f"remove ln_src {source_path}")
                    os.unlink(out_dir)
                    tqdm.tqdm.write(f"remove ln {out_dir}")
                elif os.path.isdir(out_dir):
                    shutil.rmtree(out_dir)
                    tqdm.tqdm.write(f"remove dir {out_dir}")
                else:
                    raise NotImplementedError
            if args.save_mode == "direct":
                os.makedirs(out_dir, exist_ok=True)
            if args.save_mode == "ln":
                src_path = os.path.join(args.map_dir, parent_dir, relative_out_path)
                os.makedirs(src_path, exist_ok=True)
                os.symlink(src_path, out_dir)

def overlap_boxes(box1:boxType, box2:boxType):
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

def get_border(image, expand_distance = 40):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 阈值操作将图像二值化
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours_len = len(contours)
    if contours_len == 0:
        return False, None

    boxes:list[boxType] = []

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

def save_image(file_path, image):
    if image is None:
        return
    
    def create_dir(dir:str):
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
    png_list = [os.path.basename(png_path) for png_path in glob.glob(os.path.join(raw_path,"*.png"))]

    target_size = cv2.imread(os.path.join(raw_path, png_list[0])).shape[:2]

    total_border_time = 0
    total_crop_time = 0
    total_time = 0

    for png_path in tqdm.tqdm(png_list, desc=f"{seq_name}"):
        raw_image = cv2.imread(os.path.join(raw_path, png_path))
        
        l_image = cv2.imread(os.path.join(l_path, png_path))
        r_image = cv2.imread(os.path.join(r_path, png_path))
        o_image = cv2.imread(os.path.join(o_path, png_path))

        border_start = time.time()

        flag_l, l_border = get_border(l_image)
        flag_r, r_border = get_border(r_image)
        flag_o, o_border = get_border(o_image)

        border_end = time.time()
        border_interval = border_end - border_start

        if not (flag_o and flag_l and flag_r):
            unormal_names = [name for name,flag,border in zip(["l", "r", "o"], [flag_l, flag_r, flag_o], [l_border, r_border, o_border]) if not flag and border is not None]
            if len(unormal_names) > 0:
                tqdm.tqdm.write(f"{seq_name}.{png_path} seems unormal in {unormal_names} with multi boxes detected while no overlapping")

        crop_start = time.time()

        crop(raw_image, l_border)
        crop(raw_image, r_border)
        crop(raw_image, o_border)
        crop(l_image, l_border)
        crop(r_image, r_border)
        crop(o_image, o_border)

        crop_end = time.time()
        crop_interval = crop_end - crop_start
        total = border_interval + crop_interval

        total_border_time += border_interval
        total_crop_time += crop_interval
        total_time += total

        if args.save_mode == "no":
            continue

        save_image(os.path.join(l_path, "../crop_image", png_path), crop(raw_image, l_border))
        save_image(os.path.join(r_path, "../crop_image", png_path), crop(raw_image, r_border))
        save_image(os.path.join(o_path, "../crop_image", png_path), crop(raw_image, o_border))

        save_image(os.path.join(l_path, "../crop_mask", png_path), crop(l_image, l_border))
        save_image(os.path.join(r_path, "../crop_mask", png_path), crop(r_image, r_border))
        save_image(os.path.join(o_path, "../crop_mask", png_path), crop(o_image, o_border))

    print(f"{seq_name} total border time({total_border_time/total_time*100:.2f}%): {total_border_time}, total crop time({total_crop_time/total_time*100:.2f}%): {total_crop_time}")