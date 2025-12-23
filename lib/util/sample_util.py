import torch
import numpy as np
from scipy.stats import truncnorm


def sample_beta(batch_size=1, sampling_method='normal', range_limit=4.0, device='cpu'):
    """
    对SMPL模型的beta参数进行采样，返回1x10的PyTorch张量。

    参数:
        batch_size (int): 采样样本数量，默认为1。
        sampling_method (str): 采样方法，'uniform'（均匀采样）或'normal'（正态分布采样）。默认为'normal'。
        range_limit (float): beta参数的范围限制，默认为3.0（即[-3, 3]）。
        device (torch.device): 计算设备 (e.g., 'cuda' or 'cpu')

    返回:
        torch.Tensor: 形状为(batch_size, 10)的beta参数张量。

    异常:
        ValueError: 如果sampling_method不是'uniform'或'normal'。
    """
    beta_dim = 10  # SMPL beta参数维度
    
    if sampling_method == 'uniform':
        # 均匀分布采样
        beta = np.random.uniform(low=-range_limit, high=range_limit, size=(batch_size, beta_dim))
    elif sampling_method == 'normal':
        # 正态分布采样
        beta = np.random.normal(loc=0, scale=2, size=(batch_size, beta_dim))
        beta = np.clip(beta, -range_limit, range_limit)  # 限制在[-range_limit, range_limit]
    else:
        raise ValueError("sampling_method must be 'uniform' or 'normal'")
    
    # 转换为PyTorch张量
    beta_tensor = torch.tensor(beta, dtype=torch.float32).to(device)
    
    return beta_tensor


def sample_transl4pp(batch_size, device):
    """
    为 SMPL 模型生成全局平移参数 (transl)，与数据集分布相似。

    参数：
        batch_size (int): 采样数量
        device (torch.device): 计算设备 (e.g., 'cuda' or 'cpu')

    返回：
        transl (torch.Tensor): 形状为 (batch_size, 3) 的全局平移参数 [X, Y, Z]
    """
    # X 坐标：均匀分布在 [0.45, 0.85]
    x_min, x_max = 0.45, 0.85
    x = torch.rand(batch_size, 1) * (x_max - x_min) + x_min  # 均匀分布采样
    
    # Y 坐标：均匀分布在 [1.05, 1.45]
    y_min, y_max = 1.05, 1.45
    y = torch.rand(batch_size, 1) * (y_max - y_min) + y_min  # 均匀分布采样
    
    # 卧姿对应
    # Z 坐标：截断正态分布，均值 0.08，标准差 0.03，范围 [-0.02, 0.24]
    z_mean, z_std = 0.08, 0.03
    z_min, z_max = -0.02, 0.24
    # 计算截断正态分布的标准化边界
    a, b = (z_min - z_mean) / z_std, (z_max - z_mean) / z_std
    # 使用 scipy 的 truncnorm 生成截断正态分布采样
    z = truncnorm.rvs(a, b, loc=z_mean, scale=z_std, size=(batch_size, 1))
    z = torch.tensor(z, dtype=torch.float32)
    
    # # 站姿适应
    # # Z 坐标：均匀分布在 [0.75, 0.85]
    # z_min, z_max = 0.75, 0.85
    # z = torch.rand(batch_size, 1) * (z_max - z_min) + z_min  # 均匀分布采样
    
    # 组合 X, Y, Z
    transl = torch.cat([x, y, z], dim=1).to(device)
    return transl



