use ndarray::{Array, Array1, Array3, Axis, RemoveAxis};

use crate::layers::Layer;

pub trait Softmax {
    fn softmax(&mut self, axis: usize, temperature: Option<f32>);
}

impl<D> Softmax for Array<f32, D>
where
    D: ndarray::Dimension + RemoveAxis,
{
    fn softmax(&mut self, dim: usize, temperature: Option<f32>) {
        let temp = temperature.unwrap_or(1.0);
        for mut axis in self.axis_iter_mut(ndarray::Axis(dim)) {
            let local_max = axis.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            // Subtract by local max for numerical stability
            let sum_exp: f32 = axis.iter().map(|&v| ((v - local_max) / temp).exp()).sum();
            for v in axis.iter_mut() {
                *v = ((*v - local_max) / temp).exp() / sum_exp;
            }
        }
    }
}

/// Normalizes the input across the last dimension using Layer Normalization to achieve zero mean and unit variance.
/// Formula: output = (input - mean) / sqrt(var + epsilon)
pub struct LayerNormLayer {
    // Learnable parameters for the layer to tweak the normalization result
    scale: Array1<f32>,
    shift: Array1<f32>,
}

impl LayerNormLayer {
    const EPSILON: f32 = 1e-5;

    pub fn new(dim: usize) -> Self {
        Self {
            scale: Array::ones(dim),
            shift: Array::zeros(dim),
        }
    }
}

impl Layer for LayerNormLayer {
    fn forward(&self, input: &Array3<f32>) -> Array3<f32> {
        let mut output = input.clone();

        for mut sample in output.axis_iter_mut(Axis(0)) {
            let feature_axis = Axis(1);

            for mut feature_vector in sample.axis_iter_mut(feature_axis) {
                let mean = feature_vector.mean().unwrap_or(0.0);
                let var = feature_vector.var(0.0);

                for elem in feature_vector.iter_mut() {
                    *elem = (*elem - mean) / (var + Self::EPSILON).sqrt();
                }
            }
        }

        &output * &self.scale + &self.shift
    }
}
