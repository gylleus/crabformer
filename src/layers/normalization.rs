use ndarray::{Array, Array1, Array3, Axis, RemoveAxis};
use serde::{Deserialize, Serialize};

use crate::{
    errors::ModelError,
    layers::{Layer, LayerCacheParam, ZeroGrad},
    metrics::TrainingMetricsHandle,
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
#[derive(Serialize, Deserialize)]
pub struct LayerNormLayer {
    // Learnable parameters for the layer to tweak the normalization result
    pub scale: Array1<f32>,
    pub shift: Array1<f32>,

    // Gradients for learnable parameters
    #[serde(skip)]
    scale_grad: LayerCacheParam<Array1<f32>>,
    #[serde(skip)]
    shift_grad: LayerCacheParam<Array1<f32>>,
    #[serde(skip)]
    last_input: LayerCacheParam<Array3<f32>>,
    #[serde(skip)]
    last_normalized: LayerCacheParam<Array3<f32>>,
    training: bool,
    name: String,
}

impl LayerNormLayer {
    const EPSILON: f32 = 1e-5;

    pub fn new(dim: usize, name: Option<String>) -> Self {
        let name = name.unwrap_or("LayerNormLayer".into());
        Self {
            scale: Array::ones(dim),
            shift: Array::zeros(dim),
            scale_grad: LayerCacheParam::new(format!("{}::scale_grad", name)),
            shift_grad: LayerCacheParam::new(format!("{}::shift_grad", name)),
            last_input: LayerCacheParam::new(format!("{}::last_input", name)),
            last_normalized: LayerCacheParam::new(format!("{}::last_normalized", name)),
            training: false,
            name,
        }
    }

    pub fn get_params(&mut self) -> Vec<crate::adamw::ParamHandle> {
        vec![
            crate::adamw::ParamHandle::Array1 {
                key: self.scale_grad.id(),
                data: &mut self.scale,
                grad: &self.scale_grad,
            },
            crate::adamw::ParamHandle::Array1 {
                key: self.shift_grad.id(),
                data: &mut self.shift,
                grad: &self.shift_grad,
            },
        ]
    }
}

impl Layer for LayerNormLayer {
    type Input = Array3<f32>;
    type Output = Array3<f32>;

    fn name(&self) -> &str {
        &self.name
    }

    fn forward(&self, input: &Self::Input) -> Self::Output {
        let mut output = input.clone();
        let (batch_size, seq_len, dim) = input.dim();

        // Normalize each token (each position in batch x seq) across its features (dim)
        for b in 0..batch_size {
            for s in 0..seq_len {
                // Get the feature vector for this specific token position
                let mut feature_vector = output.slice_mut(ndarray::s![b, s, ..]);

                // Compute mean and variance across features for this token
                let mean = feature_vector.mean().unwrap_or(0.0);
                let var = feature_vector.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(0.0);
                let std = (var + Self::EPSILON).sqrt();

                // Normalize: (x - mean) / std
                for elem in feature_vector.iter_mut() {
                    *elem = (*elem - mean) / std;
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
        {
            let mut guard = self.scale_grad.mut_ref();
            if guard.is_none() {
                *guard = Some(Array1::zeros(dim));
            }
        }
        {
            let mut guard = self.shift_grad.mut_ref();
            if guard.is_none() {
                *guard = Some(Array1::zeros(dim));
            }
        }

        // Gradient w.r.t. scale: sum over batch and sequence dimensions (vectorized)
        // d_scale = sum(grad_output * normalized)
        {
            let mut guard = self.scale_grad.mut_ref();
            let sg = guard.as_mut().unwrap();
            // Vectorized: sum across batch and sequence axes
            *sg = &*sg + &(grad_output * &*normalized).sum_axis(Axis(0)).sum_axis(Axis(0));
        }

        // Gradient w.r.t. shift: sum over batch and sequence dimensions (vectorized)
        // d_shift = sum(grad_output)
        {
            let mut guard = self.shift_grad.mut_ref();
            let shift_g = guard.as_mut().unwrap();
            // Vectorized: sum across batch and sequence axes
            *shift_g = &*shift_g + &grad_output.sum_axis(Axis(0)).sum_axis(Axis(0));
        }

        // Gradient w.r.t. normalized input (vectorized)
        // Broadcasting: grad_output * scale across the last dimension
        let grad_normalized = grad_output * &self.scale;

        // Gradient w.r.t. input (backprop through normalization)
        let mut grad_input = Array3::zeros((batch_size, seq_len, dim));

        for b in 0..batch_size {
            for s in 0..seq_len {
                // Extract the feature vector for this position
                let feature_vec = input.slice(ndarray::s![b, s, ..]);
                let grad_norm_vec = grad_normalized.slice(ndarray::s![b, s, ..]);

                let mean = feature_vec.mean().unwrap_or(0.0);
                let var = feature_vec.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(0.0);
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

    fn set_train(&mut self, _metrics_handle: TrainingMetricsHandle) {
        self.training = true;
    }

    fn set_eval(&mut self) {
        self.training = false;
        self.last_input.clear();
        self.last_normalized.clear();
    }
}

impl ZeroGrad for LayerNormLayer {
    fn zero_grad(&mut self) {
        if let Some(sg) = self.scale_grad.mut_ref().as_mut() {
            sg.fill(0.0);
        }
        if let Some(shift_g) = self.shift_grad.mut_ref().as_mut() {
            shift_g.fill(0.0);
        }
    }
}
