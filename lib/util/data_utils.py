import os
import torch

from config.static_args import PP_POSE_STATS_FILE, T_POSE_STATS_FILE

def load_pose_stats(device, dataset='pp'):
    """加载已保存的姿态统计量。"""
    if dataset == "pp":
        file_path = PP_POSE_STATS_FILE
    else:
        file_path = T_POSE_STATS_FILE


    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到统计文件: {file_path}，请先运行计算函数。")
        
    stats = torch.load(file_path, map_location=device)
    return stats['mean'], stats['std']

def normalize_pose(pose_raw, pose_mean, pose_std):
    """Z-Score 归一化: (X - mu) / sigma"""
    return (pose_raw - pose_mean) / pose_std

def unnormalize_pose(pose_norm, pose_mean, pose_std):
    """Z-Score 反归一化: X * sigma + mu"""
    return pose_norm * pose_std + pose_mean
