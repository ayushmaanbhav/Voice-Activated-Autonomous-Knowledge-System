//! Flow Matching ODE Solver for IndicF5
//!
//! Implements the continuous normalizing flow (CNF) sampling process
//! used in F5-TTS. Uses Euler ODE solver with Sway sampling schedule.
//!
//! # Flow Matching Overview
//!
//! Flow matching trains a model to predict the velocity field v(x, t) that
//! transforms noise into data. Sampling is done by solving the ODE:
//!
//! dx/dt = v(x_t, t)
//!
//! From t=1 (noise) to t=0 (data).

#[cfg(feature = "candle")]
use candle_core::{DType, Device, Result, Tensor};

#[cfg(feature = "candle")]
use super::config::FlowMatchingConfig;
#[cfg(feature = "candle")]
use super::dit::DiTBackbone;

/// Flow Matching sampler with Sway schedule
#[cfg(feature = "candle")]
pub struct FlowMatcher {
    config: FlowMatchingConfig,
}

#[cfg(feature = "candle")]
impl FlowMatcher {
    pub fn new(config: FlowMatchingConfig) -> Self {
        Self { config }
    }

    /// Compute the Sway sampling schedule
    ///
    /// Sway sampling adjusts the timestep schedule to improve sample quality.
    /// The transformation is: t' = t + sway_coef * t * (1 - t)
    ///
    /// This pushes more compute towards the middle of the trajectory where
    /// denoising is most critical.
    fn sway_schedule(&self, t: f32) -> f32 {
        t + self.config.sway_coef * t * (1.0 - t)
    }

    /// Generate timesteps for the ODE integration
    fn get_timesteps(&self, device: &Device) -> Result<Vec<f32>> {
        let num_steps = self.config.num_steps;
        let mut timesteps = Vec::with_capacity(num_steps + 1);

        for i in 0..=num_steps {
            let t = self.config.t_max - (i as f32 / num_steps as f32) * (self.config.t_max - self.config.t_min);
            let t_sway = self.sway_schedule(t);
            timesteps.push(t_sway);
        }

        Ok(timesteps)
    }

    /// Sample from the model using Euler ODE integration
    ///
    /// Args:
    ///   model: DiT backbone for velocity prediction
    ///   text_tokens: [batch, text_len] - input phoneme tokens
    ///   ref_mel: [batch, ref_len, n_mels] - reference mel spectrogram
    ///   target_len: Length of mel to generate
    ///   device: Target device
    ///
    /// Returns:
    ///   [batch, target_len, n_mels] - generated mel spectrogram
    pub fn sample(
        &self,
        model: &DiTBackbone,
        text_tokens: &Tensor,
        ref_mel: &Tensor,
        target_len: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let batch_size = text_tokens.dim(0)?;
        let n_mels = model.config().n_mels;

        // Initialize with noise at t=1
        let mut x = Tensor::randn(0.0f32, 1.0f32, (batch_size, target_len, n_mels), device)?;

        // Get timesteps
        let timesteps = self.get_timesteps(device)?;

        // Euler integration from t=1 to t=0
        for i in 0..timesteps.len() - 1 {
            let t = timesteps[i];
            let t_next = timesteps[i + 1];
            let dt = t_next - t;

            // Create timestep tensor
            let t_tensor = Tensor::from_vec(vec![t; batch_size], (batch_size,), device)?;

            // Predict velocity
            let v = model.forward(text_tokens, &x, &t_tensor, None)?;

            // Apply classifier-free guidance if enabled
            let v = if self.config.cfg_strength > 0.0 {
                self.apply_cfg(model, &v, text_tokens, &x, &t_tensor)?
            } else {
                v
            };

            // Euler step: x = x + dt * v
            let dt_tensor = Tensor::new(dt, device)?;
            x = x.broadcast_add(&v.broadcast_mul(&dt_tensor)?)?;
        }

        // Extract the generated portion (exclude reference)
        let ref_len = ref_mel.dim(1)?;
        if target_len > ref_len {
            x.narrow(1, ref_len, target_len - ref_len)
        } else {
            Ok(x)
        }
    }

    /// Sample with mask for in-context learning
    ///
    /// Args:
    ///   model: DiT backbone
    ///   text_tokens: [batch, text_len] - phoneme tokens
    ///   ref_mel: [batch, ref_len, n_mels] - reference audio mel
    ///   mask: [batch, total_len] - binary mask (1 = generate, 0 = keep reference)
    ///   device: Target device
    pub fn sample_with_mask(
        &self,
        model: &DiTBackbone,
        text_tokens: &Tensor,
        ref_mel: &Tensor,
        mask: &Tensor,
        device: &Device,
    ) -> Result<Tensor> {
        let batch_size = text_tokens.dim(0)?;
        let total_len = mask.dim(1)?;
        let n_mels = model.config().n_mels;

        // Initialize with noise where mask=1, reference where mask=0
        let noise = Tensor::randn(0.0f32, 1.0f32, (batch_size, total_len, n_mels), device)?;

        // Expand mask for broadcasting: [batch, len, 1]
        let mask_expanded = mask.unsqueeze(2)?;

        // Pad reference mel to total_len if needed
        let ref_len = ref_mel.dim(1)?;
        let ref_padded = if ref_len < total_len {
            let padding = Tensor::zeros((batch_size, total_len - ref_len, n_mels), DType::F32, device)?;
            Tensor::cat(&[ref_mel.clone(), padding], 1)?
        } else {
            ref_mel.narrow(1, 0, total_len)?
        };

        // x = mask * noise + (1 - mask) * ref
        let ones = Tensor::ones_like(&mask_expanded)?;
        let inv_mask = ones.broadcast_sub(&mask_expanded)?;
        let mut x = mask_expanded.broadcast_mul(&noise)?.broadcast_add(
            &inv_mask.broadcast_mul(&ref_padded)?
        )?;

        // Get timesteps
        let timesteps = self.get_timesteps(device)?;

        // Euler integration
        for i in 0..timesteps.len() - 1 {
            let t = timesteps[i];
            let t_next = timesteps[i + 1];
            let dt = t_next - t;

            let t_tensor = Tensor::from_vec(vec![t; batch_size], (batch_size,), device)?;

            // Predict velocity
            let v = model.forward(text_tokens, &x, &t_tensor, None)?;

            // Apply CFG
            let v = if self.config.cfg_strength > 0.0 {
                self.apply_cfg(model, &v, text_tokens, &x, &t_tensor)?
            } else {
                v
            };

            // Euler step only for masked positions
            let dt_tensor = Tensor::new(dt, device)?;
            let x_update = v.broadcast_mul(&dt_tensor)?;
            let x_update_masked = x_update.broadcast_mul(&mask_expanded)?;
            x = x.broadcast_add(&x_update_masked)?;

            // Keep reference positions fixed
            x = mask_expanded.broadcast_mul(&x)?.broadcast_add(
                &inv_mask.broadcast_mul(&ref_padded)?
            )?;
        }

        Ok(x)
    }

    /// Apply classifier-free guidance
    ///
    /// v_guided = v_cond + cfg_strength * (v_cond - v_uncond)
    fn apply_cfg(
        &self,
        model: &DiTBackbone,
        v_cond: &Tensor,
        text_tokens: &Tensor,
        x: &Tensor,
        t: &Tensor,
    ) -> Result<Tensor> {
        // For CFG, we'd need to compute unconditional prediction
        // This is a simplified version - full CFG would mask text tokens
        let batch_size = text_tokens.dim(0)?;
        let text_len = text_tokens.dim(1)?;

        // Create "empty" text tokens (zeros)
        let empty_tokens = Tensor::zeros((batch_size, text_len), DType::U32, text_tokens.device())?;

        // Unconditional prediction
        let v_uncond = model.forward(&empty_tokens, x, t, None)?;

        // Guided prediction: v_cond + cfg * (v_cond - v_uncond)
        let diff = v_cond.sub(&v_uncond)?;
        let cfg_scale = Tensor::new(self.config.cfg_strength, v_cond.device())?;
        v_cond.broadcast_add(&diff.broadcast_mul(&cfg_scale)?)
    }

    /// Get configuration
    pub fn config(&self) -> &FlowMatchingConfig {
        &self.config
    }

    /// Set number of integration steps
    pub fn set_num_steps(&mut self, num_steps: usize) {
        self.config.num_steps = num_steps;
    }

    /// Set CFG strength
    pub fn set_cfg_strength(&mut self, strength: f32) {
        self.config.cfg_strength = strength;
    }
}

/// Noise schedule utilities
#[cfg(feature = "candle")]
pub struct NoiseSchedule;

#[cfg(feature = "candle")]
impl NoiseSchedule {
    /// Linear interpolation between noise and data
    ///
    /// x_t = t * x_0 + (1 - t) * noise
    pub fn interpolate(x_0: &Tensor, noise: &Tensor, t: f32, device: &Device) -> Result<Tensor> {
        let t_tensor = Tensor::new(t, device)?;
        let one_minus_t = Tensor::new(1.0 - t, device)?;

        let weighted_x = x_0.broadcast_mul(&t_tensor)?;
        let weighted_noise = noise.broadcast_mul(&one_minus_t)?;

        weighted_x.broadcast_add(&weighted_noise)
    }

    /// Compute target velocity for training
    ///
    /// v = x_0 - noise
    pub fn compute_velocity(x_0: &Tensor, noise: &Tensor) -> Result<Tensor> {
        x_0.sub(noise)
    }

    /// Sample random timesteps uniformly in [0, 1]
    pub fn sample_timesteps(batch_size: usize, device: &Device) -> Result<Tensor> {
        Tensor::rand(0.0f32, 1.0f32, (batch_size,), device)
    }
}

// Non-Candle stubs
#[cfg(not(feature = "candle"))]
pub struct FlowMatcher;

#[cfg(not(feature = "candle"))]
pub struct NoiseSchedule;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sway_schedule() {
        let config = FlowMatchingConfig::default();

        #[cfg(feature = "candle")]
        {
            let matcher = FlowMatcher::new(config);

            // At t=0 and t=1, sway should have no effect
            assert!((matcher.sway_schedule(0.0) - 0.0).abs() < 1e-6);
            assert!((matcher.sway_schedule(1.0) - 1.0).abs() < 1e-6);

            // At t=0.5, sway should shift by sway_coef * 0.25
            let t_mid = matcher.sway_schedule(0.5);
            let expected = 0.5 + config.sway_coef * 0.25;
            assert!((t_mid - expected).abs() < 1e-6);
        }
    }
}
