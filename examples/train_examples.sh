#!/bin/bash
# Example training commands for the motion generation project

echo "Motion Generation Training Examples"
echo "===================================="
echo ""

# Example 1: Train with default convolutional mid structure
echo "Example 1: Train with convolutional mid layer (default)"
echo "Command:"
echo "python train.py \\"
echo "    --experiment_name conv_baseline \\"
echo "    --epochs 50 \\"
echo "    --batch_size 64 \\"
echo "    --learning_rate 1e-4 \\"
echo "    --mid_structure conv"
echo ""

# Example 2: Train with attention mid structure
echo "Example 2: Train with attention mid layer"
echo "Command:"
echo "python train.py \\"
echo "    --experiment_name attn_baseline \\"
echo "    --epochs 50 \\"
echo "    --batch_size 64 \\"
echo "    --learning_rate 1e-4 \\"
echo "    --mid_structure attention \\"
echo "    --mid_num_heads 8"
echo ""

# Example 3: Train with attention and custom architecture
echo "Example 3: Train with larger attention architecture"
echo "Command:"
echo "python train.py \\"
echo "    --experiment_name attn_large \\"
echo "    --base_channels 256 \\"
echo "    --channel_multipliers 1 2 4 8 \\"
echo "    --mid_structure attention \\"
echo "    --mid_num_heads 16 \\"
echo "    --time_emb_dim 512"
echo ""

# Example 4: Resume training
echo "Example 4: Resume training from checkpoint"
echo "Command:"
echo "python train.py \\"
echo "    --experiment_name resumed_training \\"
echo "    --resume outputs/conv_baseline_20231114_120000/checkpoints/latest.pt"
echo ""

# Example 5: Quick test run
echo "Example 5: Quick test run (1 epoch, small batch)"
echo "Command:"
echo "python train.py \\"
echo "    --experiment_name test_run \\"
echo "    --epochs 1 \\"
echo "    --batch_size 8 \\"
echo "    --mid_structure attention \\"
echo "    --save_interval 1"
echo ""

echo "Output Structure:"
echo "================="
echo "outputs/"
echo "└── {experiment_name}_{timestamp}/"
echo "    ├── config.json           # Saved hyperparameters"
echo "    ├── checkpoints/          # Model checkpoints"
echo "    │   ├── latest.pt"
echo "    │   ├── best.pt"
echo "    │   └── epoch_*.pt"
echo "    ├── logs/                 # Training logs"
echo "    │   └── training.log"
echo "    ├── tensorboard/          # TensorBoard logs"
echo "    └── samples/              # Generated samples"
