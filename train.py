# motion_generation/train.py

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from lib.dataset import PressurePoseDataset, InBedPressureDataset
from lib.model.unet1d import UNet1D
from lib.model.ddpm import DDPM
from lib.model.flow_matching import FlowMatching
from lib.util.data_utils import unnormalize_pose
import os
import argparse
import logging
import json
from datetime import datetime
from tqdm import tqdm
from config.static_args import TIP_PATH

def setup_logger(log_dir, name='Training'):
    """Setup logger with both file and console handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    logger.handlers = []

    # File handler
    log_file = os.path.join(log_dir, 'training.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, is_best=False, save_interval=10):
    """Save model checkpoint with training state"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    # Save latest checkpoint
    latest_path = os.path.join(checkpoint_dir, 'latest.pt')
    torch.save(checkpoint, latest_path)

    # Save epoch checkpoint
    if (epoch + 1) % save_interval == 0:  # Save every save_interval epochs
        epoch_path = os.path.join(checkpoint_dir, f'epoch_{epoch:04d}.pt')
        torch.save(checkpoint, epoch_path)

    # Save best checkpoint
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best.pt')
        torch.save(checkpoint, best_path)

def save_config(args, save_dir):
    """Save training configuration to JSON file"""
    config = vars(args).copy()
    # Convert any non-serializable types
    for key, value in config.items():
        if isinstance(value, torch.device):
            config[key] = str(value)
        elif isinstance(value, list):
            config[key] = value

    config_path = os.path.join(save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    return config_path

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load model checkpoint and restore training state"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))

    return epoch, loss

def train_diffusion_model(args):
    # Setup directories with better organization
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{args.exp_name}_{timestamp}" if args.exp_name else timestamp
    if args.curr_sid < 0:
        exp_name = exp_name.replace('tip', f'tip_{0-args.curr_sid}')

    # import pdb; pdb.set_trace()

    # Create organized directory structure
    # run_dir: contains all outputs for this run
    run_dir = os.path.join(args.output_dir, exp_name)
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    log_dir = os.path.join(run_dir, 'logs')
    tensorboard_dir = os.path.join(run_dir, 'tensorboard')
    sample_dir = os.path.join(run_dir, 'samples')

    # Create all directories
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    # Setup logger and tensorboard
    logger_name = f"{args.model_type.upper()}_Training"
    logger = setup_logger(log_dir, name=logger_name)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # Device setup
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Output directory: {run_dir}")

    # Save configuration
    config_path = save_config(args, run_dir)
    logger.info(f"Configuration saved to: {config_path}")

    # Dataset initialization
    logger.info(f"Loading datasets {args.dataset}...")
    if args.dataset == "pp":
        train_data = PressurePoseDataset(split='train', device=device)
    else: 
        # cfgs = {
        #     'dataset_path': TIP_PATH,
        #     'dataset_mode': 'unseen_group',
        #     'curr_fold': 1,
        #     'normalize': False,
        #     'device': device,
        # }
        cfgs = {
            'dataset_path': TIP_PATH,
            'dataset_mode': 'unseen_subject',
            'curr_fold': args.curr_sid,
            'normalize': False,
            'device': device,
        }
        train_data = InBedPressureDataset(cfgs=cfgs, mode='all')
    logger.info(f"Loaded train data: #{0-cfgs['curr_fold']}# {len(train_data)} samples")

    # DataLoader
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        # pin_memory=True if device.type == 'cuda' else False
    )

    # Model initialization
    logger.info(f"Initializing {args.model_type.upper()} model...")
    logger.info(f"UNet mid structure: {args.mid_structure}")
    if args.mid_structure == 'attention':
        logger.info(f"Attention heads: {args.mid_num_heads}")

    unet = UNet1D(
        pose_dim=72,
        base_channels=args.base_channels,
        channel_multipliers=args.channel_multipliers,
        time_emb_dim=args.time_emb_dim,
        mid_structure=args.mid_structure,
        mid_num_heads=args.mid_num_heads
    ).to(device)

    # Create either DDPM or FlowMatching model
    if args.model_type == 'ddpm':
        model = DDPM(
            model=unet,
            timesteps=args.timesteps
        ).to(device)
        logger.info(f"DDPM initialized with {args.timesteps} timesteps")
    elif args.model_type == 'flow_matching':
        model = FlowMatching(
            model=unet,
            sigma=args.flow_sigma
        ).to(device)
        logger.info(f"FlowMatching initialized with sigma={args.flow_sigma}")
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}. Choose 'ddpm' or 'flow_matching'")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.learning_rate * 0.01
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    best_loss = float('inf')

    if args.resume:
        if os.path.exists(args.resume):
            logger.info(f"Resuming from checkpoint: {args.resume}")
            start_epoch, prev_loss = load_checkpoint(args.resume, model, optimizer)
            start_epoch += 1
            best_loss = prev_loss
            logger.info(f"Resumed from epoch {start_epoch}, previous loss: {prev_loss:.6f}")
        else:
            logger.warning(f"Checkpoint not found: {args.resume}")

    # Log hyperparameters
    logger.info("Hyperparameters:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")

    # Training loop
    logger.info("Starting training...")
    global_step = 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0

        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for step, batch in enumerate(pbar):
            # Get normalized pose data
            pose_norm = batch['pose_norm'].float().to(device)  # (B, 72)

            # Forward pass
            loss = model(pose_norm)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            # Update metrics
            epoch_loss += loss.item()
            global_step += 1

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

            # Log to tensorboard
            if global_step % args.log_interval == 0:
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)

        # Epoch statistics
        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch [{epoch+1}/{args.epochs}] Average Loss: {avg_epoch_loss:.6f}")

        writer.add_scalar('train/epoch_loss', avg_epoch_loss, epoch)

        # Update learning rate
        scheduler.step()

        # Save checkpoint
        is_best = avg_epoch_loss < best_loss
        if is_best:
            best_loss = avg_epoch_loss
            logger.info(f"New best model! Loss: {best_loss:.6f}")

        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1 or is_best:
            save_checkpoint(model, optimizer, epoch, avg_epoch_loss, checkpoint_dir, is_best, args.save_interval)
            logger.info(f"Checkpoint saved at epoch {epoch+1}")

    logger.info("Training completed!")
    logger.info(f"Best loss: {best_loss:.6f}")
    logger.info(f"Checkpoints saved in: {checkpoint_dir}")

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DDPM or Flow Matching for SMPL Pose Generation")

    # Model type selection
    parser.add_argument('--model_type', type=str, default='ddpm', choices=['ddpm', 'flow_matching'],
                        help='Type of generative model to use: ddpm or flow_matching')
    parser.add_argument('--dataset', type=str, default="pp", choices=["tip", "pp"], 
                        help="Dataset for training")
    parser.add_argument('--curr_sid', type=int, default=1, 
                        help='Current fold for cross-validation')

    # Training parameters
    parser.add_argument('--device', type=str, default='cuda:2' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--epochs', type=int, default=50, 
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay for optimizer')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping threshold (0 to disable)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of DataLoader workers')

    # Model parameters
    parser.add_argument('--timesteps', type=int, default=1000,
                        help='Number of diffusion timesteps (for DDPM)')
    parser.add_argument('--flow_sigma', type=float, default=0.0,
                        help='Standard deviation for conditional flow matching (for FlowMatching, 0.0 for standard flow matching)')
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

    # Logging and saving
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save all outputs (models, logs, configs, samples)')
    parser.add_argument('--exp_name', type=str, default='',
                        help='Experiment name (timestamp will be appended)')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Steps between logging to tensorboard')
    parser.add_argument('--save_interval', type=int, default=999,
                        help='Epochs between saving checkpoint')

    # Resume training
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()

    train_diffusion_model(args)

