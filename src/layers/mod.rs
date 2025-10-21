use ndarray::{Array, Array2, Array3};
use rand::{Rng, rngs::StdRng};

pub mod activation;
pub mod dropout;
pub mod embedding;
pub mod linear;
pub mod multi_head_attention;
pub mod normalization;
pub mod self_attention;
pub mod transformer_block;

/// Initialize an array with Xavier/Glorot initialization.
/// This should keep variance of activations and gradients roughly the same across layers, to keep training more stable.
pub fn xavier_initialized_array(fan_in: usize, fan_out: usize, rng: &mut StdRng) -> Array2<f32> {
    let limit = (6.0 / (fan_in as f32 + fan_out as f32)).sqrt();

    Array::from_shape_fn((fan_in, fan_out), |_| rng.random_range(-limit..limit))
}

pub trait Layer {
    fn forward(&self, input: &Array3<f32>) -> Array3<f32>;
}
