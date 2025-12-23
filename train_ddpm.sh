python train.py \
    --model_type ddpm \
    --epoch 50 \
    --batch_size 64 \
    --timesteps 1000 \
    --base_channels 128 \
    --channel_multipliers 1 2 4 \
    --time_emb_dim 128 \
    --mid_structure conv \



