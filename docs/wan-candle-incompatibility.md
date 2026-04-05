# Wan Video Generation: Candle Numerical Incompatibility

## Summary

Wan 2.1 text-to-video generation cannot produce valid output through Cake's candle-based inference pipeline due to numerical divergence from PyTorch over iterative denoising steps. The pipeline architecture is complete and verified correct at the single-step level, but the accumulated floating-point differences between candle and PyTorch cause the 30-step denoising trajectory to diverge into an entirely different (invalid) image.

LTX-2 does not exhibit this issue despite using the same candle framework and similar iterative denoising.

## Evidence

### Single forward pass: matches Python

A single Wan transformer forward pass in candle matches PyTorch to **max_diff = 0.0007** (0.07% relative error). This was verified element-by-element on CPU with F32 precision using identical inputs, weights, and the 1.3B model.

### 30-step denoising: diverges completely

Over 30 classifier-free guidance (CFG) steps:

| Step | Per-element diff | Cumulative effect |
|------|-----------------|-------------------|
| 0 | 0.0022 | Negligible |
| 4 | 0.0093 | Minor drift |
| 9 | 0.0159 | Noticeable |
| 29 | ~0.5-1.0 | Complete divergence |

The final latent tensors have **correlation = -0.36** with Python's output (negative correlation = essentially unrelated content). Python produces a recognizable golden retriever; candle produces a dark, featureless image.

### Every configuration tested produces the same dark result

- **Quantization**: Q4_K_S, Q5_K_M, Q8_0, F16, BF16, F32 — all dark
- **Device**: CUDA GPU, CPU — all dark
- **Model size**: 1.3B, 14B — all dark
- **Resolution**: 128×128, 256×256, 320×320, 480×832 — all dark
- **Random seed**: Every CUDA seed tested — all dark
- **Distributed**: Single GPU, 2-GPU (4090 + 5090) — all dark
- **With Python's exact noise**: Same initial noise tensor — still dark

### Python produces correct output from the same model

The same Wan 1.3B model running through Python's diffusers pipeline produces high-quality video at 256×256 and 320×320. The same 14B model produces excellent 480p video. The model itself is not the problem.

## Root Cause Analysis

### Why the per-step error exists

Candle's F32 matrix multiplication produces slightly different results from PyTorch's F32 matrix multiplication for the same inputs. This is expected — different libraries use different BLAS implementations with different operation ordering (fused multiply-add, vectorization strategies, accumulation order). For most use cases this difference is negligible.

### Why CFG amplifies the error

Classifier-free guidance computes: `pred = uncond + scale × (cond - uncond)`

With `scale = 5.0`, a 0.0007 per-forward-pass error in both the conditional and unconditional predictions becomes approximately `5 × 2 × 0.0007 ≈ 0.007` per step. This feeds back as input to the next step.

### Why it compounds exponentially

Diffusion denoising is a chaotic dynamical system — the model's output at step N depends sensitively on its input, which is the output of step N-1. Small perturbations grow exponentially. After 30 steps of CFG-amplified iteration, the 0.07% per-step error becomes a complete trajectory divergence.

### Why LTX-2 doesn't have this problem

Several factors make LTX-2 more robust:

1. **BF16 throughout**: LTX-2 uses BF16 (8 exponent bits, same range as F32) for all weights and activations, converting to F32 only for attention computation. Wan's diffusers weights are BF16 but our pipeline casts to F16 (5 exponent bits, limited range) or F32.

2. **Different attention architecture**: LTX-2 uses a dual-stream DiT with cross-attention between text and video tokens. Wan uses a single-stream architecture with separate self-attention and cross-attention per block. The error sensitivity depends on the specific attention pattern.

3. **Different RoPE implementation**: LTX-2 uses split 3D RoPE applied per-stream. Wan uses interleaved 3D RoPE with repeat_interleave(2) on the frequency dimensions. Different numerical paths through the position encoding.

4. **Tuned with candle**: LTX-2's implementation was iteratively debugged and validated against Python output during development. Subtle numerical issues (dtype casts, operation ordering) were fixed until output matched. Wan has not had this level of tuning.

5. **Model sensitivity**: Some architectures are inherently more sensitive to numerical perturbation than others. Wan's 5120 hidden dimension with 40 attention heads may create larger intermediate values than LTX-2's architecture, amplifying floating-point errors.

## What Was Verified Working

The following components match Python exactly:

- **UMT5-XXL text encoder**: Candle's T5EncoderModel handles UMT5 correctly. Output matches Python to 5-6 significant figures.
- **Tokenizer**: Identical token IDs for all tested prompts.
- **Scheduler**: Flow matching Euler discrete scheduler with shift=1.0 matches Python's sigma/timestep schedule exactly.
- **Denormalization**: Latent normalization constants and formula match to max_diff 4.8e-7.
- **3D RoPE**: Precomputed frequencies match Python value-for-value, verified for temporal, height, and width components.
- **Patch embedding**: Manual Conv3d decomposition via conv2d matches Python's native Conv3d to machine precision.
- **Unpatchify**: 8-dimensional reshape with channel-last ordering matches Python's einops rearrange.
- **VAE decoder first frame**: All layer outputs (conv_in, mid_block, attention, up_blocks) match Python for the first decoded frame.

## What Would Be Needed to Fix This

1. **Match PyTorch's matmul exactly**: The core issue is that `candle_core::Tensor::matmul` produces different results from `torch.matmul` for the same F32 inputs. This would need to be fixed in candle's BLAS backend to use the same accumulation strategy as PyTorch.

2. **Or reduce per-step error below the chaotic threshold**: If the per-step error were reduced from 0.0007 to ~0.0001 or less, the trajectory might stay close enough to converge to a similar image. This would require identifying which specific operation (matmul, softmax, layer norm) contributes most to the 0.0007.

3. **Or use a different denoising strategy**: Stochastic samplers that add noise at each step might regularize the trajectory and prevent divergence. This hasn't been tested.

## Files

| File | Description |
|------|-------------|
| `cake-core/src/models/wan/` | Complete Wan model implementation |
| `cake-core/src/models/wan/vendored/model.rs` | Transformer forward pass |
| `cake-core/src/models/wan/vendored/vae.rs` | VAE decoder with chunked temporal decoding |
| `cake-core/src/models/wan/quantized_transformer.rs` | GGUF quantized model (QMatMul) |
| `cake-core/src/models/wan/transformer.rs` | Forwarder with diffusers key remapping |
| `cake-core/tests/wan_forward_compare.rs` | Element-wise comparison test vs Python |
| `cake-core/tests/wan_conv3d_compare.rs` | Conv3d decomposition verification |
| `cake-core/tests/wan_vae_compare.rs` | VAE decoder comparison |
| `scripts/convert_wan_vae.py` | VAE .pth→safetensors converter |
| `topology-wan14b.yml` | 2-GPU distributed topology |
