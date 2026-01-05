# motion_generation/visualize.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
import smplx
from lib.util.sample_util import sample_beta, sample_transl4pp
from config.static_args import SMPL_MODEL


# SMPL kinematic tree (parent joint indices)
SMPL_PARENTS = [
    -1,  # 0: Pelvis (root)
    0,   # 1: L_Hip
    0,   # 2: R_Hip
    0,   # 3: Spine1
    1,   # 4: L_Knee
    2,   # 5: R_Knee
    3,   # 6: Spine2
    4,   # 7: L_Ankle
    5,   # 8: R_Ankle
    6,   # 9: Spine3
    7,   # 10: L_Foot
    8,   # 11: R_Foot
    9,   # 12: Neck
    9,   # 13: L_Collar
    9,   # 14: R_Collar
    12,  # 15: Head
    13,  # 16: L_Shoulder
    14,  # 17: R_Shoulder
    16,  # 18: L_Elbow
    17,  # 19: R_Elbow
    18,  # 20: L_Wrist
    19,  # 21: R_Wrist
    20,  # 22: L_Hand
    21,  # 23: R_Hand
]

# Joint names for better understanding
SMPL_JOINT_NAMES = [
    'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2',
    'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot', 'Neck',
    'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
    'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'
]


def load_samples(sample_path):
    """Load generated samples from .pt file"""
    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"Sample file not found: {sample_path}")

    samples = torch.load(sample_path)
    print(f"Loaded samples with shape: {samples.shape}")
    return samples


def smpl_pose_to_joints(pose_params, betas=None, transl=None, device='cpu'):
    """
    Convert SMPL rotation parameters to joint positions using forward kinematics

    Args:
        pose_params: (B, 72) or (72,) SMPL rotation parameters (24 joints × 3 axis-angle)
        betas: (B, 10) or (10,) body shape parameters (optional)
        transl: (B, 3) or (3,) global translation (optional)
        device: computation device

    Returns:
        joints: (B, 24, 3) or (24, 3) joint positions
    """
    # Handle single sample
    squeeze_output = False
    if pose_params.dim() == 1:
        pose_params = pose_params.unsqueeze(0)
        squeeze_output = True
        if betas is not None and betas.dim() == 1:
            betas = betas.unsqueeze(0)
        if transl is not None and transl.dim() == 1:
            transl = transl.unsqueeze(0)

    batch_size = pose_params.shape[0]

    # Default betas and transl if not provided
    if betas is None:
        betas = torch.zeros(batch_size, 10, device=device)
    if transl is None:
        transl = torch.zeros(batch_size, 3, device=device)

    # Reshape pose to (B, 24, 3)
    pose_reshaped = pose_params.reshape(batch_size, 24, 3)

    # Split into global_orient and body_pose
    global_orient = pose_reshaped[:, 0:1, :]  # (B, 1, 3) - root rotation
    body_pose = pose_reshaped[:, 1:, :]       # (B, 23, 3) - body joint rotations

    # Create SMPL model
    smpl_model = smplx.create(
        SMPL_MODEL,
        model_type='smpl',
        gender='neutral',
        batch_size=batch_size
    ).to(device)

    # Forward pass to get joint positions
    output = smpl_model(
        body_pose=body_pose,
        global_orient=global_orient,
        betas=betas,
        transl=transl,
        return_verts=False
    )

    joints = output.joints[:, :24, :]  # (B, 24, 3)

    if squeeze_output:
        joints = joints.squeeze(0)

    return joints


def visualize_skeleton_3d(joints, title="SMPL Skeleton", save_path=None):
    """
    Visualize SMPL skeleton in 3D

    Args:
        joints: (24, 3) joint positions
        title: plot title
        save_path: path to save figure (optional)
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    joints_np = joints.cpu().numpy() if torch.is_tensor(joints) else joints

    # Plot joints
    ax.scatter(joints_np[:, 0], joints_np[:, 1], joints_np[:, 2],
               c='red', marker='o', s=50, label='Joints')

    # Plot bones
    for i, parent in enumerate(SMPL_PARENTS):
        if parent >= 0:
            x = [joints_np[parent, 0], joints_np[i, 0]]
            y = [joints_np[parent, 1], joints_np[i, 1]]
            z = [joints_np[parent, 2], joints_np[i, 2]]
            ax.plot(x, y, z, 'b-', linewidth=2)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()

    # Set equal aspect ratio
    max_range = np.array([
        joints_np[:, 0].max() - joints_np[:, 0].min(),
        joints_np[:, 1].max() - joints_np[:, 1].min(),
        joints_np[:, 2].max() - joints_np[:, 2].min()
    ]).max() / 2.0

    mid_x = (joints_np[:, 0].max() + joints_np[:, 0].min()) * 0.5
    mid_y = (joints_np[:, 1].max() + joints_np[:, 1].min()) * 0.5
    mid_z = (joints_np[:, 2].max() + joints_np[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved skeleton visualization to: {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_smpl_mesh(pose_params, betas, transl, title="SMPL Mesh", save_path=None, device='cpu'):
    """
    Visualize SMPL mesh with vertices from rotation parameters

    Args:
        pose_params: (72,) SMPL rotation parameters (24 joints × 3 axis-angle)
        betas: (10,) body shape parameters
        transl: (3,) global translation
        title: plot title
        save_path: path to save figure (optional)
        device: computation device
    """
    # Ensure single sample
    if pose_params.dim() == 2:
        pose_params = pose_params.squeeze(0)
    if betas.dim() == 2:
        betas = betas.squeeze(0)
    if transl.dim() == 2:
        transl = transl.squeeze(0)

    # Reshape pose to (1, 24, 3)
    pose_reshaped = pose_params.reshape(1, 24, 3)
    global_orient = pose_reshaped[:, 0:1, :]  # (1, 1, 3)
    body_pose = pose_reshaped[:, 1:, :]       # (1, 23, 3)

    # Create SMPL model
    smpl_model = smplx.create(
        SMPL_MODEL,
        model_type='smpl',
        gender='neutral',
        batch_size=1
    ).to(device)

    # Forward pass to get vertices
    output = smpl_model(
        body_pose=body_pose,
        global_orient=global_orient,
        betas=betas.unsqueeze(0),
        transl=transl.unsqueeze(0),
        return_verts=True
    )

    vertices = output.vertices[0].cpu().numpy()  # (6890, 3)

    # for tip
    vertices[:, 1] = 1.80 - vertices[:, 1]
    vertices[:, 2] = -vertices[:, 2]

    faces = smpl_model.faces

    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot mesh
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                    triangles=faces, cmap='viridis', alpha=0.7,
                    edgecolor='none', shade=True)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # Set equal aspect ratio
    max_range = np.array([
        vertices[:, 0].max() - vertices[:, 0].min(),
        vertices[:, 1].max() - vertices[:, 1].min(),
        vertices[:, 2].max() - vertices[:, 2].min()
    ]).max() / 2.0

    mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
    mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
    mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved mesh visualization to: {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_multiple_samples(samples, num_samples=4, method='skeleton',
                               betas=None, transl=None, save_dir=None, device='cpu'):
    """
    Visualize multiple samples in a grid

    Args:
        samples: (N, 72) SMPL rotation parameters (24 joints × 3 axis-angle)
        num_samples: number of samples to visualize
        method: 'skeleton' or 'mesh'
        betas: (N, 10) body shape parameters
        transl: (N, 3) global translation
        save_dir: directory to save figures
        device: computation device
    """
    num_samples = min(num_samples, samples.shape[0])

    if method == 'skeleton':
        # Convert SMPL rotation parameters to joint positions
        joints = smpl_pose_to_joints(
            samples[:num_samples],
            betas=betas[:num_samples] if betas is not None else None,
            transl=transl[:num_samples] if transl is not None else None,
            device=device
        )

        # Create grid visualization
        cols = min(4, num_samples)
        rows = (num_samples + cols - 1) // cols

        fig = plt.figure(figsize=(5*cols, 5*rows))

        for i in range(num_samples):
            ax = fig.add_subplot(rows, cols, i+1, projection='3d')

            joints_i = joints[i].cpu().numpy() if torch.is_tensor(joints) else joints[i]

            # Plot joints
            ax.scatter(joints_i[:, 0], joints_i[:, 1], joints_i[:, 2],
                      c='red', marker='o', s=30)

            # Plot bones
            for j, parent in enumerate(SMPL_PARENTS):
                if parent >= 0:
                    x = [joints_i[parent, 0], joints_i[j, 0]]
                    y = [joints_i[parent, 1], joints_i[j, 1]]
                    z = [joints_i[parent, 2], joints_i[j, 2]]
                    ax.plot(x, y, z, 'b-', linewidth=1.5)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Sample {i+1}')

            # Set equal aspect ratio
            max_range = np.array([
                joints_i[:, 0].max() - joints_i[:, 0].min(),
                joints_i[:, 1].max() - joints_i[:, 1].min(),
                joints_i[:, 2].max() - joints_i[:, 2].min()
            ]).max() / 2.0

            mid_x = (joints_i[:, 0].max() + joints_i[:, 0].min()) * 0.5
            mid_y = (joints_i[:, 1].max() + joints_i[:, 1].min()) * 0.5
            mid_z = (joints_i[:, 2].max() + joints_i[:, 2].min()) * 0.5

            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'skeleton_grid.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved skeleton grid to: {save_path}")
        else:
            plt.show()

        plt.close()

    elif method == 'mesh':
        # Visualize meshes one by one using rotation parameters directly
        for i in range(num_samples):
            title = f'SMPL Mesh - Sample {i+1}'
            save_path = None

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'mesh_sample_{i+1:03d}.png')

            visualize_smpl_mesh(
                samples[i],
                betas[i] if betas is not None else torch.zeros(10, device=device),
                transl[i] if transl is not None else torch.zeros(3, device=device),
                title=title,
                save_path=save_path,
                device=device
            )


def main(args):
    """Main visualization function"""
    print("="*60)
    print("SMPL Pose Visualization Tool")
    print("="*60)

    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load samples
    print(f"\nLoading samples from: {args.sample_path}")
    samples = load_samples(args.sample_path)

    # Move samples to device
    if not torch.is_tensor(samples):
        samples = torch.tensor(samples, dtype=torch.float32)
    samples = samples.to(device)

    # Select sample indices
    if args.indices:
        indices = args.indices
    else:
        indices = list(range(min(args.num_samples, len(samples))))

    print(f"Visualizing {len(indices)} samples: {indices}")

    # Sample betas and transl if needed
    if args.method == 'mesh' or args.use_betas_transl:
        print("\nSampling betas and transl parameters...")
        betas = sample_beta(
            batch_size=len(samples),
            sampling_method='normal',
            range_limit=4.0,
            device=device
        )
        transl = sample_transl4pp(
            batch_size=len(samples),
            device=device
        )
        print(f"Betas shape: {betas.shape}")
        print(f"Transl shape: {transl.shape}")
    else:
        betas = None
        transl = None

    # Visualization
    print(f"\nVisualization method: {args.method}")

    if args.method == 'skeleton':
        if len(indices) == 1:
            # Single skeleton
            joints = smpl_pose_to_joints(
                samples[indices[0]],
                betas=betas[indices[0]] if betas is not None else None,
                transl=transl[indices[0]] if transl is not None else None,
                device=device
            )
            save_path = os.path.join(args.output_dir, f'skeleton_sample_{indices[0]:03d}.png') if args.output_dir else None
            visualize_skeleton_3d(joints, title=f"SMPL Skeleton - Sample {indices[0]}", save_path=save_path)
        else:
            # Multiple skeletons in grid
            selected_samples = samples[indices]
            selected_betas = betas[indices] if betas is not None else None
            selected_transl = transl[indices] if transl is not None else None
            visualize_multiple_samples(
                selected_samples,
                num_samples=len(indices),
                method='skeleton',
                betas=selected_betas,
                transl=selected_transl,
                save_dir=args.output_dir,
                device=device
            )

    elif args.method == 'mesh':
        selected_samples = samples[indices]
        selected_betas = betas[indices]
        selected_transl = transl[indices]
        visualize_multiple_samples(
            selected_samples,
            num_samples=len(indices),
            method='mesh',
            betas=selected_betas,
            transl=selected_transl,
            save_dir=args.output_dir,
            device=device
        )

    print("\nVisualization completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize SMPL pose samples")

    # Input
    parser.add_argument('--sample_path', type=str, required=True,
                        help='Path to saved samples (.pt file)')
    parser.add_argument('--method', type=str, default='mesh', choices=['skeleton', 'mesh'],
                        help='Visualization method: skeleton or mesh')

    # Sample selection
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Number of samples to visualize')
    parser.add_argument('--indices', nargs='+', type=int, default=None,
                        help='Specific sample indices to visualize')

    # SMPL parameters
    parser.add_argument('--use_betas_transl', action='store_true',
                        help='Sample and use betas/transl for skeleton visualization (affects body shape and position)')

    # Output
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save visualizations (if not specified, will display)')

    # Device
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')

    args = parser.parse_args()

    main(args)
