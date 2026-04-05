use candle_core::{DType, Tensor};

type Result<T> = anyhow::Result<T>;

/// FP32 Layer Norm without learnable parameters.
/// Computes: (x - mean) / sqrt(var + eps)
/// Returns F32 to avoid premature truncation (caller decides final dtype).
pub fn fp32_layer_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let x = x.to_dtype(DType::F32)?;
    let mean = x.mean_keepdim(candle_core::D::Minus1)?;
    let x = x.broadcast_sub(&mean)?;
    let var = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    let out = x.broadcast_div(&(var + eps)?.sqrt()?)?;
    Ok(out)
}

/// Apply AdaLN modulation: norm(x) * (1 + scale) + shift.
/// Computes entirely in F32, returns in original dtype.
pub fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor, eps: f64) -> Result<Tensor> {
    let in_dtype = x.dtype();
    let x_norm = fp32_layer_norm(x, eps)?; // already F32
    let scale = scale.to_dtype(DType::F32)?;
    let shift = shift.to_dtype(DType::F32)?;
    let ones = Tensor::ones_like(&scale)?;
    let out = x_norm
        .broadcast_mul(&(scale + ones)?)?
        .broadcast_add(&shift)?;
    Ok(out.to_dtype(in_dtype)?)
}
