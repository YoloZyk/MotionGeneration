# motion_generation/sample.py

import torch
from lib.model.unet1d import UNet1D
from lib.model.ddpm import DDPM
from lib.model.flow_matching import FlowMatching
from lib.dataset.pressurepose import PressurePoseDataset
from lib.util.data_utils import unnormalize_pose
import argparse
import os
import logging
from tqdm import tqdm
import numpy as np
import json

def setup_logger(name='Sampling'):
    """Setup console logger"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger

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
    """Load trained model from checkpoint"""
    # Initialize model
    unet = UNet1D(
        pose_dim=72,
        base_channels=args.base_channels,
        channel_multipliers=args.channel_multipliers,
        time_emb_dim=args.time_emb_dim,
        mid_structure=getattr(args, 'mid_structure', 'conv'),
        mid_num_heads=getattr(args, 'mid_num_heads', 4)
    ).to(device)

    # Create either DDPM or FlowMatching model based on model_type
    model_type = getattr(args, 'model_type', 'ddpm')
    if model_type == 'ddpm':
        model = DDPM(
            model=unet,
            timesteps=args.timesteps
        ).to(device)
    elif model_type == 'flow_matching':
        model = FlowMatching(
            model=unet,
            sigma=getattr(args, 'flow_sigma', 0.0)
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'ddpm' or 'flow_matching'")

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

def sample_poses(args):
    """Generate pose samples from trained model (DDPM or Flow Matching)"""
    # Setup logger based on model type
    model_type = getattr(args, 'model_type', 'ddpm')
    logger_name = f"{model_type.upper()}_Sampling"
    logger = setup_logger(name=logger_name)

    # Device setup
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Check if checkpoint exists
    if not os.path.exists(args.ckpt):
        logger.error(f"Checkpoint not found: {args.ckpt}")
        return

    # Detect run directory from checkpoint path
    run_dir = detect_run_dir(args.ckpt)
    logger.info(f"Detected run directory: {run_dir}")

    # Load config if available and not overridden
    config = load_config(run_dir)
    if config:
        logger.info("Found config.json in run directory")
        # Update args with config values if not explicitly provided
        if not hasattr(args, '_explicit_params'):
            # Use config values for model parameters
            for key in ['base_channels', 'channel_multipliers', 'time_emb_dim',
                       'timesteps', 'mid_structure', 'mid_num_heads', 'model_type', 'flow_sigma']:
                if key in config and not getattr(args, f'{key}_override', False):
                    setattr(args, key, config[key])
                    logger.info(f"  Using {key} from config: {config[key]}")
    else:
        logger.info("No config.json found, using command-line arguments")

    # Setup output directory - save to samples subdirectory in run_dir
    if args.output_dir is None:
        output_dir = os.path.join(run_dir, 'samples')
    else:
        # If user explicitly provided output_dir, use it
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Samples will be saved to: {output_dir}")

    # Load dataset to get normalization statistics
    logger.info("Loading dataset for normalization statistics...")
    dataset = PressurePoseDataset(split='train', device='cpu')
    pose_mean = dataset.pose_mean.to(device)
    pose_std = dataset.pose_std.to(device)

    # Load model
    logger.info(f"Loading model from: {args.ckpt}")
    model, epoch, loss = load_model(args.ckpt, device, args)
    logger.info(f"Model loaded - Epoch: {epoch}, Loss: {loss}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    # Generate samples
    logger.info(f"Generating {args.num_samples} samples...")
    logger.info(f"Using chunked saving: {args.chunk_size} samples per file")

    all_samples_norm = []
    all_samples_raw = []
    chunk_idx = 0
    total_saved = 0

    # Generate in batches
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Sampling"):
            # Calculate batch size for this iteration
            current_batch_size = min(args.batch_size, args.num_samples - i * args.batch_size)

            # Generate normalized samples
            if args.save_intermediates and i == 0:
                # Save intermediates only for the first batch
                if model_type == 'ddpm':
                    samples_norm, intermediates = model.sample(
                        sample_shape=(current_batch_size, 72),
                        device=device,
                        return_intermediates=True,
                        save_interval=args.intermediate_interval
                    )
                else:  # flow_matching
                    samples_norm, intermediates = model.sample(
                        sample_shape=(current_batch_size, 72),
                        device=device,
                        num_steps=args.flow_num_steps,
                        method=args.flow_solver,
                        return_intermediates=True,
                        save_interval=args.intermediate_interval,
                        verbose=False
                    )
                # Save intermediates to file
                intermediate_path = os.path.join(output_dir, 'intermediates.pt')
                torch.save(intermediates, intermediate_path)
                logger.info(f"Saved {len(intermediates)} intermediate states to: {intermediate_path}")
            else:
                if model_type == 'ddpm':
                    samples_norm = model.sample(
                        sample_shape=(current_batch_size, 72),
                        device=device
                    )
                else:  # flow_matching
                    samples_norm = model.sample(
                        sample_shape=(current_batch_size, 72),
                        device=device,
                        num_steps=args.flow_num_steps,
                        method=args.flow_solver,
                        verbose=False
                    )

            # Unnormalize samples
            samples_raw = unnormalize_pose(samples_norm, pose_mean, pose_std)

            all_samples_norm.append(samples_norm.cpu())
            all_samples_raw.append(samples_raw.cpu())

            # Check if we need to save a chunk
            current_count = sum(s.shape[0] for s in all_samples_norm)
            if current_count >= args.chunk_size or i == num_batches - 1:
                # Concatenate accumulated samples
                # chunk_samples_norm = torch.cat(all_samples_norm, dim=0)
                chunk_samples_raw = torch.cat(all_samples_raw, dim=0)

                # Save chunk
                # output_path_norm = os.path.join(output_dir, f'samples_normalized_{chunk_idx:05d}.pt')
                output_path_raw = os.path.join(output_dir, f'samples_raw_{chunk_idx:05d}.pt')

                # torch.save(chunk_samples_norm, output_path_norm)
                torch.save(chunk_samples_raw, output_path_raw)

                logger.info(f"Chunk {chunk_idx}: Saved {chunk_samples_raw.shape[0]} samples")
                # logger.info(f"  Normalized: {output_path_norm}")
                logger.info(f"  Raw: {output_path_raw}")

                # Save as numpy arrays if requested
                if args.save_numpy:
                    # output_path_norm_npy = os.path.join(output_dir, f'samples_normalized_{chunk_idx:05d}.npy')
                    output_path_raw_npy = os.path.join(output_dir, f'samples_raw_{chunk_idx:05d}.npy')

                    # np.save(output_path_norm_npy, chunk_samples_norm.numpy())
                    np.save(output_path_raw_npy, chunk_samples_raw.numpy())

                    # logger.info(f"  Normalized (numpy): {output_path_norm_npy}")
                    logger.info(f"  Raw (numpy): {output_path_raw_npy}")

                # Print chunk statistics
                logger.info(f"  Chunk statistics:")
                # logger.info(f"    Normalized - Mean: {chunk_samples_norm.mean():.6f}, Std: {chunk_samples_norm.std():.6f}")
                logger.info(f"    Raw - Mean: {chunk_samples_raw.mean():.6f}, Std: {chunk_samples_raw.std():.6f}")

                total_saved += chunk_samples_raw.shape[0]

                # Clear lists and increment chunk index
                all_samples_norm = []
                all_samples_raw = []
                chunk_idx += 1

    logger.info(f"\nTotal samples saved: {total_saved} across {chunk_idx} chunks")
    logger.info(f"Samples saved in: {output_dir}")
    logger.info("\nSampling completed successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sample poses from trained DDPM or Flow Matching model")

    # Model checkpoint
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to model checkpoint')

    # Model type
    parser.add_argument('--model_type', type=str, default=None, choices=['ddpm', 'flow_matching'],
                        help='Type of model (auto-detected from config.json if available)')

    # Sampling parameters
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for sampling')
    parser.add_argument('--chunk_size', type=int, default=2000,
                        help='Number of samples per chunk file (to avoid large files)')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')

    # Model parameters (must match training configuration)
    # These can be auto-loaded from config.json if available in the run directory
    parser.add_argument('--timesteps', type=int, default=1000,
                        help='Number of diffusion timesteps (for DDPM)')
    parser.add_argument('--flow_sigma', type=float, default=0.0,
                        help='Standard deviation for conditional flow matching (for FlowMatching)')
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
                        help='Directory to save generated samples (default: auto-detect {run_dir}/samples)')
    parser.add_argument('--save_numpy', action='store_true',
                        help='Also save samples as numpy arrays')

    # Flow matching specific parameters
    parser.add_argument('--flow_num_steps', type=int, default=100,
                        help='Number of ODE integration steps for flow matching (default: 100)')
    parser.add_argument('--flow_solver', type=str, default='euler',
                        choices=['euler', 'midpoint', 'rk4'],
                        help='ODE solver method for flow matching (default: euler)')

    # Intermediate states
    parser.add_argument('--save_intermediates', action='store_true',
                        help='Save intermediate states during sampling (first sample only)')
    parser.add_argument('--intermediate_interval', type=int, default=100,
                        help='Save intermediate states every N timesteps')

    args = parser.parse_args()

    sample_poses(args)
