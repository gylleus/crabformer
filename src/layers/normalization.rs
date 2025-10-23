use ndarray::{Array, Array1, Array3, Axis, RemoveAxis};

use crate::{
    errors::ModelError,
    layers::{Layer, LayerCacheParam, ZeroGrad},
};

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
/// Formula: output = scale * (input - mean) / sqrt(var + epsilon) + shift
pub struct LayerNormLayer {
    // Learnable parameters for the layer to tweak the normalization result
    pub scale: Array1<f32>,
    pub shift: Array1<f32>,
    // Gradients for learnable parameters
    pub scale_grad: Option<Array1<f32>>,
    pub shift_grad: Option<Array1<f32>>,
    // Cache for backward pass
    last_input: LayerCacheParam<Array3<f32>>,
    last_normalized: LayerCacheParam<Array3<f32>>,
    training: bool,
}

impl LayerNormLayer {
    const EPSILON: f32 = 1e-5;

    pub fn new(dim: usize) -> Self {
        Self {
            scale: Array::ones(dim),
            shift: Array::zeros(dim),
            scale_grad: None,
            shift_grad: None,
            last_input: LayerCacheParam::new("LayerNormLayer::last_input"),
            last_normalized: LayerCacheParam::new("LayerNormLayer::last_normalized"),
            training: false,
        }
    }
}

impl Layer for LayerNormLayer {
    type Input = Array3<f32>;
    type Output = Array3<f32>;

    fn forward(&self, input: &Self::Input) -> Self::Output {
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

        if self.training {
            *self.last_input.mut_ref() = Some(input.clone());
            *self.last_normalized.mut_ref() = Some(output.clone());
        }

        output * &self.scale + &self.shift
    }

    // fn backward(&mut self, grad_output: &Array3<f32>) -> Result<Array3<f32>, ModelError> {
    fn backward(&mut self, grad_output: &Self::Output) -> Result<Self::Input, ModelError> {
        let input = self.last_input.read_ref()?;
        let normalized = self.last_normalized.read_ref()?;

        let (batch_size, seq_len, dim) = input.dim();

        // Lazy init gradients
        if self.scale_grad.is_none() {
            self.scale_grad = Some(Array1::zeros(dim));
        }
        if self.shift_grad.is_none() {
            self.shift_grad = Some(Array1::zeros(dim));
        }

        // Gradient w.r.t. scale: sum over batch and sequence dimensions
        // d_scale = sum(grad_output * normalized)
        if let Some(ref mut sg) = self.scale_grad {
            for b in 0..batch_size {
                for s in 0..seq_len {
                    for d in 0..dim {
                        sg[d] += grad_output[[b, s, d]] * normalized[[b, s, d]];
                    }
                }
            }
        }

        // Gradient w.r.t. shift: sum over batch and sequence dimensions
        // d_shift = sum(grad_output)
        if let Some(ref mut shift_g) = self.shift_grad {
            for b in 0..batch_size {
                for s in 0..seq_len {
                    for d in 0..dim {
                        shift_g[d] += grad_output[[b, s, d]];
                    }
                }
            }
        }

        // Gradient w.r.t. normalized input
        let mut grad_normalized = Array3::zeros((batch_size, seq_len, dim));
        for b in 0..batch_size {
            for s in 0..seq_len {
                for d in 0..dim {
                    grad_normalized[[b, s, d]] = grad_output[[b, s, d]] * self.scale[d];
                }
            }
        }

        // Gradient w.r.t. input (backprop through normalization)
        let mut grad_input = Array3::zeros((batch_size, seq_len, dim));

        for b in 0..batch_size {
            for s in 0..seq_len {
                // Extract the feature vector for this position
                let feature_vec = input.slice(ndarray::s![b, s, ..]);
                let grad_norm_vec = grad_normalized.slice(ndarray::s![b, s, ..]);

                let mean = feature_vec.mean().unwrap_or(0.0);
                let var = feature_vec.var(0.0);
                let std = (var + Self::EPSILON).sqrt();

                let n = dim as f32;

                // Backprop through normalization: y = (x - mean) / std
                // dy/dx = (1/std) * (I - 1/n - (x - mean)(x - mean)^T / (n * var))
                // Simplified computation:
                let sum_grad = grad_norm_vec.sum();
                let sum_grad_normalized =
                    (&grad_norm_vec * &normalized.slice(ndarray::s![b, s, ..])).sum();

                for d in 0..dim {
                    let dx = (grad_norm_vec[d]
                        - sum_grad / n
                        - normalized[[b, s, d]] * sum_grad_normalized / n)
                        / std;
                    grad_input[[b, s, d]] = dx;
                }
            }
        }

        Ok(grad_input.into())
    }

    fn set_train(&mut self) {
        self.training = true;
    }

    fn set_eval(&mut self) {
        self.training = false;
        self.last_input.clear();
        self.last_normalized.clear();
    }
}

impl LayerNormLayer {
    /// Zero out accumulated gradients
    pub fn zero_grad(&mut self) {
        if let Some(ref mut sg) = self.scale_grad {
            sg.fill(0.0);
        }
        if let Some(ref mut shift_g) = self.shift_grad {
            shift_g.fill(0.0);
        }
    }
}

impl ZeroGrad for LayerNormLayer {
    fn zero_grad(&mut self) {}
}
