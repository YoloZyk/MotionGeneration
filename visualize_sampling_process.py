# motion_generation/visualize_sampling_process.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
import smplx
from tqdm import tqdm
import json

from lib.model.unet1d import UNet1D
from lib.model.ddpm import DDPM
from lib.dataset.pressurepose import PressurePoseDataset
from lib.util.data_utils import unnormalize_pose
from lib.util.sample_util import sample_beta, sample_transl4pp
from config.static_args import SMPL_MODEL


def detect_run_dir(checkpoint_path):
    """
    Detect the run directory from checkpoint path.
    Assumes structure: {run_dir}/checkpoints/{checkpoint_name}.pt
    """
    abs_path = os.path.abspath(checkpoint_path)
    # If checkpoint is in a 'checkpoints' directory, go up one level
    if 'checkpoints' in abs_path:
        checkpoint_dir = os.path.dirname(abs_path)
        if os.path.basename(checkpoint_dir) == 'checkpoints':
            return os.path.dirname(checkpoint_dir)
    # Otherwise, use the directory containing the checkpoint
    return os.path.dirname(abs_path)


def load_config(run_dir):
    """Load training configuration from run directory"""
    config_path = os.path.join(run_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    return None


def load_model(checkpoint_path, device, args):
    """Load trained DDPM model from checkpoint"""
    # Initialize model
    unet = UNet1D(
        pose_dim=72,
        base_channels=args.base_channels,
        channel_multipliers=args.channel_multipliers,
        time_emb_dim=args.time_emb_dim,
        mid_structure=getattr(args, 'mid_structure', 'conv'),
        mid_num_heads=getattr(args, 'mid_num_heads', 4)
    ).to(device)

    model = DDPM(
        model=unet,
        timesteps=args.timesteps
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle both full checkpoint and state_dict only
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        loss = checkpoint.get('loss', 'unknown')
    else:
        model.load_state_dict(checkpoint)
        epoch = 'unknown'
        loss = 'unknown'

    model.eval()

    return model, epoch, loss


def smpl_pose_to_mesh(pose_params, betas, transl, device='cpu'):
    """
    Convert SMPL rotation parameters to mesh vertices

    Args:
        pose_params: (72,) SMPL rotation parameters
        betas: (10,) body shape parameters
        transl: (3,) global translation
        device: computation device

    Returns:
        vertices: (6890, 3) mesh vertices
        faces: triangle faces
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
    faces = smpl_model.faces

    return vertices, faces


def visualize_sampling_process(intermediates, betas, transl, output_dir, device='cpu',
                               show_timesteps=None):
    """
    Visualize the DDPM sampling process by showing intermediate mesh states

    Args:
        intermediates: List[(timestep, x_t)] from DDPM sampling
        betas: (10,) body shape parameters
        transl: (3,) global translation
        output_dir: directory to save visualizations
        device: computation device
        show_timesteps: specific timesteps to visualize (if None, visualize all)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset for unnormalization
    print("Loading dataset for normalization statistics...")
    dataset = PressurePoseDataset(split='train', device='cpu')
    pose_mean = dataset.pose_mean.to(device)
    pose_std = dataset.pose_std.to(device)

    # Filter timesteps if needed
    if show_timesteps is not None:
        intermediates = [(t, x) for t, x in intermediates if t in show_timesteps]

    print(f"Visualizing {len(intermediates)} intermediate states...")

    # Create individual mesh images
    for i, (timestep, x_norm) in enumerate(tqdm(intermediates, desc="Generating meshes")):
        # Unnormalize pose
        x_norm = x_norm.to(device)
        pose_raw = unnormalize_pose(x_norm, pose_mean, pose_std)

        # Get mesh
        vertices, faces = smpl_pose_to_mesh(pose_raw[0], betas, transl, device)

        # Create 3D plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot mesh
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                       triangles=faces, cmap='viridis', alpha=0.8,
                       edgecolor='none', shade=True)

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Sampling Process - Timestep {timestep}', fontsize=16)

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

        # Set consistent view angle
        ax.view_init(elev=20, azim=45)

        plt.show()
        # Save
        # save_path = os.path.join(output_dir, f'step_{i:04d}_t{timestep:04d}.png')
        # plt.savefig(save_path, dpi=150, bbox_inches='tight')
        # plt.close()

    print(f"Saved {len(intermediates)} intermediate meshes to: {output_dir}")

    # Create a comparison grid
    print("Creating comparison grid...")
    create_comparison_grid(intermediates, betas, transl, output_dir, device,
                          pose_mean, pose_std)


def create_comparison_grid(intermediates, betas, transl, output_dir, device,
                           pose_mean, pose_std):
    """Create a grid showing multiple intermediate states side by side"""

    # Select evenly spaced timesteps for the grid
    num_show = min(8, len(intermediates))
    indices = np.linspace(0, len(intermediates) - 1, num_show, dtype=int)
    selected = [intermediates[i] for i in indices]

    # Create grid
    cols = 4
    rows = (num_show + cols - 1) // cols
    fig = plt.figure(figsize=(5*cols, 5*rows))

    for i, (timestep, x_norm) in enumerate(selected):
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')

        # Unnormalize pose
        x_norm = x_norm.to(device)
        pose_raw = unnormalize_pose(x_norm, pose_mean, pose_std)

        # Get mesh
        vertices, faces = smpl_pose_to_mesh(pose_raw[0], betas, transl, device)

        # Plot mesh
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                       triangles=faces, cmap='viridis', alpha=0.8,
                       edgecolor='none', shade=True)

        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.set_zlabel('Z', fontsize=8)
        ax.set_title(f't={timestep}', fontsize=12)

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

        ax.view_init(elev=20, azim=45)
        ax.tick_params(labelsize=6)

    plt.suptitle('DDPM Sampling Process: From Noise to Pose', fontsize=16, y=0.995)
    plt.tight_layout()

    grid_path = os.path.join(output_dir, 'sampling_process_grid.png')
    plt.savefig(grid_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Saved comparison grid to: {grid_path}")


def main(args):
    """Main function"""
    print("="*60)
    print("DDPM Sampling Process Visualization")
    print("="*60)

    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Check if checkpoint exists
    if not os.path.exists(args.ckpt):
        print(f"Error: Checkpoint not found: {args.ckpt}")
        return

    # Detect run directory from checkpoint path
    run_dir = detect_run_dir(args.ckpt)
    print(f"Detected run directory: {run_dir}")

    # Load config if available
    config = load_config(run_dir)
    if config:
        print("Found config.json in run directory")
        # Update args with config values if not explicitly provided
        for key in ['base_channels', 'channel_multipliers', 'time_emb_dim',
                   'timesteps', 'mid_structure', 'mid_num_heads']:
            if key in config and not getattr(args, f'{key}_override', False):
                setattr(args, key, config[key])
                print(f"  Using {key} from config: {config[key]}")
    else:
        print("No config.json found, using command-line arguments")

    # Setup output directory - save to samples/visualizations subdirectory in run_dir
    if args.output_dir is None:
        output_dir = os.path.join(run_dir, 'samples', 'visualizations')
    else:
        # If user explicitly provided output_dir, use it
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    print(f"Visualizations will be saved to: {output_dir}")

    # Load model
    print(f"\nLoading model from: {args.ckpt}")
    model, epoch, loss = load_model(args.ckpt, device, args)
    print(f"Model loaded - Epoch: {epoch}, Loss: {loss}")

    # Sample betas and transl
    print("\nSampling body shape and translation parameters...")
    betas = sample_beta(
        batch_size=1,
        sampling_method='normal',
        range_limit=4.0,
        device=device
    )[0]
    transl = sample_transl4pp(
        batch_size=1,
        device=device
    )[0]

    # Generate sample with intermediates
    print(f"\nGenerating sample with intermediate states (save_interval={args.save_interval})...")
    final_sample, intermediates = model.sample(
        sample_shape=(1, 72),
        device=device,
        return_intermediates=True,
        save_interval=args.save_interval
    )

    print(f"Generated {len(intermediates)} intermediate states")
    print(f"Timesteps: {[t for t, _ in intermediates]}")

    # Visualize sampling process
    print("\nVisualizing sampling process...")
    visualize_sampling_process(
        intermediates,
        betas,
        transl,
        output_dir,
        device,
        show_timesteps=args.show_timesteps
    )

    print("\nVisualization completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize DDPM sampling process")

    # Model checkpoint
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to model checkpoint')

    # Sampling parameters
    parser.add_argument('--save_interval', type=int, default=100,
                        help='Save intermediate states every N timesteps (smaller = more frames)')
    parser.add_argument('--show_timesteps', nargs='+', type=int, default=None,
                        help='Specific timesteps to visualize (if not specified, show all saved)')

    # Model parameters (must match training configuration)
    # These can be auto-loaded from config.json if available in the run directory
    parser.add_argument('--timesteps', type=int, default=1000,
                        help='Number of diffusion timesteps')
    parser.add_argument('--base_channels', type=int, default=128,
                        help='Base channels for UNet1D')
    parser.add_argument('--channel_multipliers', nargs='+', type=int, default=[1, 2, 4],
                        help='Channel multipliers for UNet1D layers')
    parser.add_argument('--time_emb_dim', type=int, default=256,
                        help='Time embedding dimension')
    parser.add_argument('--mid_structure', type=str, default='conv', choices=['conv', 'attention'],
                        help='Structure for UNet mid layer: conv (ResBlock1D) or attention (SelfAttention1D)')
    parser.add_argument('--mid_num_heads', type=int, default=4,
                        help='Number of attention heads for mid layer when using attention structure')

    # Output
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save visualizations (default: auto-detect {run_dir}/samples/visualizations)')

    # Device
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')

    args = parser.parse_args()

    main(args)
