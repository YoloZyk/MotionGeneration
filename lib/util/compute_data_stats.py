# motion_generation/lib/util/data_util.py
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from config.static_args import PP_POSE_STATS_FILE, T_POSE_STATS_FILE
from lib.dataset.pressurepose import PressurePoseDataset 
from lib.dataset.tip import InBedPressureDataset

def compute_pose_stats(dataset, save_path):
    """
    计算训练集上 72 维姿态参数的均值和标准差，并保存。
    """
    print("开始计算姿态参数统计量...")
    all_poses = []

    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

    # 遍历整个数据集，获取所有姿态参数
    for i, batch in enumerate(tqdm(dataloader, desc="loading")):
        body_pose = batch['smpl'][:, 10:79]
        orient = batch['smpl'][:, 79:82]

        pose = torch.cat([orient, body_pose], dim=1)
        all_poses.append(pose)

    all_poses_tensor = torch.cat(all_poses, dim=0) # (N, 72)
    
    import pdb; pdb.set_trace()

    pose_mean = all_poses_tensor.mean(dim=0, keepdim=True) # (1, 72)
    pose_std = all_poses_tensor.std(dim=0, keepdim=True)  # (1, 72)

    # 避免除以零：将极小的标准差替换为小常数 epsilon
    epsilon = 1e-6
    pose_std[pose_std < epsilon] = epsilon

    # 保存统计量
    torch.save({
        'mean': pose_mean,
        'std': pose_std
    }, save_path)
    print(f"统计量计算完毕并保存到: {save_path}")
    
    return pose_mean, pose_std

# --------------------------------------------------------------------------
# 注意：在训练前，需要单独运行一次 compute_pose_stats 来生成 stats 文件！
# 例如：
if __name__ == "__main__":
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    dataset = PressurePoseDataset(split='train', device=device)
    
    # cfgs = {
    #     'dataset_path': "/workspace/zyk/public_data/wzy_opt_dataset_w_feats",
    #     'dataset_mode': 'unseen_group',  # or 'unseen_subject'
    #     'curr_fold': 1,  # Used for 'unseen_subject' mode (1, 2, or 3)
    #     'normalize': False,
    #     'device': torch.device('cuda:1' if torch.cuda.is_available() else 'cpu'),  # or 'cuda' for GPU
    # }
    # dataset = InBedPressureDataset(cfgs, mode='train')

    compute_pose_stats(dataset, save_path=PP_POSE_STATS_FILE)


