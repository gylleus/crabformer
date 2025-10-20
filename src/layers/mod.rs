use ndarray::{Array, Array2};
use rand::{Rng, rngs::StdRng};

pub mod dropout;
pub mod embedding;
pub mod normalize;
pub mod self_attention;

pub fn xavier_initialized_array(fan_in: usize, fan_out: usize, rng: &mut StdRng) -> Array2<f32> {
    let limit = (6.0 / (fan_in as f32 + fan_out as f32)).sqrt();

    Array::from_shape_fn((fan_in, fan_out), |_| rng.random_range(-limit..limit))
}
