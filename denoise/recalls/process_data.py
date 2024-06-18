from pytorch3d.transforms.rotation_conversions import matrix_to_axis_angle

def process_pose(pose):
    return matrix_to_axis_angle(pose.reshape(-1, 3, 3)).reshape(-1, 48)

def cal_diff(x):
    # x: [Bs, C, T]
    return x[:,:,1:] - x[:,:,:-1]