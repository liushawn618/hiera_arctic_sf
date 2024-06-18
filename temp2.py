import torch
import numpy as np
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, RasterizationSettings, MeshRenderer, MeshRasterizer, SoftPhongShader, TexturesVertex
)
from pytorch3d.structures import Meshes
import matplotlib.pyplot as plt
import math

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载多个3D模型并存储在字典中
model_paths = {
    "model1": "path/to/your/model1.obj",
    "model2": "path/to/your/model2.obj"
}
meshes_dict = {name: load_objs_as_meshes([path], device=device) for name, path in model_paths.items()}

# 批量渲染设置
batch_size = 64
rows, cols = 224, 224

# 设置相机参数
def setup_viewer(cam_t):
    focal = 1000.0
    K = np.array([[focal, 0, cols / 2.0], [0, focal, rows / 2.0], [0, 0, 1]])
    Rt = np.zeros((batch_size, 3, 4))  # batch_size frames, 1 scene
    Rt[:, :, 3] = cam_t
    Rt[:, :3, :3] = np.eye(3)
    Rt[:, 1:3, :3] *= -1.0

    R = Rt[:, :3, :3]
    t = Rt[:, :, 3]

    fx = K[0, 0]
    fy = K[1, 1]
    px = K[0, 2]
    py = K[1, 2]

    R = torch.tensor(R, dtype=torch.float32, device=device)
    t = torch.tensor(t, dtype=torch.float32, device=device)
    focal_length = torch.tensor([[fx, fy]], dtype=torch.float32, device=device)
    principal_point = torch.tensor([[px, py]], dtype=torch.float32, device=device)

    cameras = FoVPerspectiveCameras(
        R=R,
        T=t,
        focal_length=focal_length,
        principal_point=principal_point,
        image_size=((rows, cols),),
        device=device
    )

    return cameras

# 示例调用
cam_t = np.random.rand(batch_size, 3) * 10  # 示例相机位置
cameras = setup_viewer(cam_t)

# 设置光栅化
raster_settings = RasterizationSettings(
    image_size=224,
    blur_radius=0.0,
    faces_per_pixel=1,
)

# 创建渲染器
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(device=device, cameras=cameras)
)

# 批量渲染
def render_batch(vertices_batch, faces_batch, model_names, rotations):
    images_list = []

    for i in range(batch_size):
        model_name = model_names[i]
        mesh = meshes_dict[model_name]

        # 获取当前帧的顶点和面
        verts = vertices_batch[i]
        faces = faces_batch[i]

        # 创建旋转矩阵
        radian = rotations[i]
        cos_angle = math.cos(radian)
        sin_angle = math.sin(radian)
        rotation_matrix = torch.tensor([
            [cos_angle, -sin_angle, 0],
            [sin_angle, cos_angle, 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=device)
        
        # 应用旋转
        rotated_verts = verts @ rotation_matrix.T

        # 创建Mesh
        mesh = Meshes(
            verts=[rotated_verts],
            faces=[faces],
            textures=TexturesVertex(verts_features=torch.ones_like(rotated_verts))
        )

        # 渲染当前模型
        images = renderer(mesh)
        images_list.append(images)

    # 合并所有图像
    all_images = torch.cat(images_list, dim=0)
    return all_images

# 示例输入
vertices_batch = [meshes_dict[name].verts_list()[0] for name in model_paths.keys()] * (batch_size // 2)
faces_batch = [meshes_dict[name].faces_list()[0] for name in model_paths.keys()] * (batch_size // 2)
vertices_batch = torch.stack(vertices_batch)
faces_batch = torch.stack(faces_batch)
model_names = ["model1", "model2"] * (batch_size // 2)  # 每帧对应的模型名称
rotations = torch.rand(batch_size) * 2 * math.pi  # 每帧的旋转角度

# 渲染批量图像
images = render_batch(vertices_batch, faces_batch, model_names, rotations)

# 显示渲染图像的第一个示例
plt.imshow(images[0, ..., :3].cpu().numpy())
plt.axis("off")
plt.show()
