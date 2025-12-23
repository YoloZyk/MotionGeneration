# UNet1D Mid Layer Structure Comparison

## Quick Reference

| Feature | Conv Structure | Attention Structure |
|---------|---------------|---------------------|
| Module Type | ResBlock1D | SelfAttention1D |
| Operation | 1D Convolution | Multi-head Self-Attention |
| Parameters (default config) | ~5.77M | ~5.24M |
| Receptive Field | Local (kernel-based) | Global (attention-based) |
| Time Complexity | O(C × L × K) | O(C × L²) |
| Space Complexity | O(1) | O(L²) for attention matrix |
| Best For | Local patterns | Long-range dependencies |

Where:
- C = number of channels
- L = sequence length
- K = kernel size

## When to Use Each Structure

### Convolutional Structure (conv)
**Use when:**
- Input sequences are short
- Patterns are primarily local
- Memory is limited
- Faster training is needed
- You want the proven baseline architecture

**Advantages:**
- Efficient for local patterns
- Lower memory footprint
- Faster computation for long sequences
- More stable training

### Attention Structure (attention)
**Use when:**
- Need to model long-range dependencies
- Global context is important
- Sequences have complex interactions
- You have sufficient memory
- Experimenting with transformer-style architectures

**Advantages:**
- Captures global dependencies
- Can learn complex relationships
- Position-aware through learned patterns
- Potentially better feature representations

## Parameter Details

### Conv Structure (ResBlock1D)
```
Components:
- 2x Conv1d layers (with BN + GELU)
- 1x Time projection (Linear)
- 1x Residual projection (if needed)
```

### Attention Structure (SelfAttention1D)
```
Components:
- 1x GroupNorm
- 1x QKV projection (Conv1d, 3x channels)
- 1x Output projection (Conv1d)
- 1x Time projection (Linear)
- Multi-head attention mechanism
```

## Training Considerations

### Memory Usage
- **Conv**: ~constant memory per layer
- **Attention**: Memory grows with L² due to attention matrix
  - For L=72 (pose dim), attention matrix is 72×72 per head
  - With 8 heads: ~41K floats per sample in attention matrix

### Computational Cost
- **Conv**: Linear in sequence length
- **Attention**: Quadratic in sequence length
  - For short sequences (L=72), this overhead is minimal
  - Still faster than many alternatives for pose data

### Hyperparameter Sensitivity
- **Conv**: More robust to hyperparameter choices
- **Attention**:
  - Number of heads affects capacity and memory
  - Ensure `mid_channels % num_heads == 0`
  - Default config: 512 channels (4×128), 8 heads = 64 dims/head

## Experimental Recommendations

### Baseline Experiments
1. Start with **conv** structure (proven baseline)
2. Train for full epochs with default settings
3. Record final loss and sample quality

### Comparison Experiments
1. Train with **attention** structure
2. Use same hyperparameters (except mid_structure/num_heads)
3. Compare:
   - Final training loss
   - Convergence speed
   - Sample quality
   - Training time
   - Memory usage

### Architecture Search
Try different `mid_num_heads` for attention:
- 4 heads: More dims per head (128), might capture richer features
- 8 heads: Balanced (64 dims/head) - **default**
- 16 heads: More diverse attention patterns (32 dims/head)

## Code Examples

### Python API
```python
from lib.model.unet1d import UNet1D

# Conv structure
model_conv = UNet1D(
    pose_dim=72,
    base_channels=128,
    mid_structure='conv'
)

# Attention structure
model_attn = UNet1D(
    pose_dim=72,
    base_channels=128,
    mid_structure='attention',
    mid_num_heads=8
)
```

### Command Line
```bash
# Conv structure
python train.py --mid_structure conv

# Attention structure
python train.py --mid_structure attention --mid_num_heads 8
```

## Expected Results

Based on the architecture:
- **Similar performance** for simple pose patterns
- **Attention may excel** at complex multi-joint interactions
- **Conv may train faster** due to lower computational cost
- **Attention uses less parameters** but more memory during forward pass

## Monitoring Tips

Check in TensorBoard:
1. Loss curves - should converge similarly
2. Learning rate schedule - same for both
3. Gradient norms - attention may have different gradient flow

Check in logs:
1. Training time per epoch
2. Memory usage (if logged)
3. Final model size on disk
