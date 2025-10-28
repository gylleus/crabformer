use ndarray::{Array1, Array2, Array3, s};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    adamw,
    errors::ModelError,
    layers::{
        Layer, LayerCacheParam, ZeroGrad,
        activation::{Gelu, gelu_derivative},
        xavier_initialized_array,
    },
    metrics::TrainingMetricsHandle,
};

#[derive(Serialize, Deserialize)]

/// 2D linear layer with optional bias
pub struct LinearLayer {
    pub weights: Array2<f32>,
    pub bias: Option<Array1<f32>>,

    #[serde(skip)]
    weight_grad: LayerCacheParam<Array2<f32>>,
    #[serde(skip)]
    bias_grad: LayerCacheParam<Array1<f32>>,
    #[serde(skip)]
    last_input: LayerCacheParam<Array3<f32>>,

    training: bool,
    name: String,
}

impl LinearLayer {
    pub fn new(dim_in: usize, dim_out: usize, name: Option<String>) -> Self {
        let name = name.unwrap_or("LinearLayer".into());
        Self {
            weights: xavier_initialized_array(dim_in, dim_out),
            bias: None,
            weight_grad: LayerCacheParam::new(format!("{}::weight_grad", name)),
            bias_grad: LayerCacheParam::new(format!("{}::bias_grad", name)),
            last_input: LayerCacheParam::new(format!("{}::last_input", name)),
            training: false,
            name,
        }
    }

    pub fn with_bias(mut self) -> Self {
        self.bias = Some(Array1::zeros(self.weights.dim().1));
        self
    }
}

impl Layer for LinearLayer {
    type Input = Array3<f32>;
    type Output = Array3<f32>;

    fn name(&self) -> &str {
        &self.name
    }

    fn forward(&self, input: &Self::Input) -> Self::Output {
        let (batch_size, seq_len, _dim_in) = input.dim();
        let dim_out = self.weights.dim().1;

        // Cache the input for backward pass
        if self.training {
            *self.last_input.mut_ref() = Some(input.clone());
        }

        // Parallelize across batch dimension
        let batch_results: Vec<Array2<f32>> = (0..batch_size)
            .into_par_iter()
            .map(|b| {
                let batch_input = input.slice(s![b, .., ..]);
                let mut batch_output = batch_input.dot(&self.weights);
                if let Some(bias) = &self.bias {
                    batch_output += bias;
                }
                batch_output
            })
            .collect();

        // Combine results back into 3D array
        let mut output = Array3::<f32>::zeros((batch_size, seq_len, dim_out));
        for (b, result) in batch_results.into_iter().enumerate() {
            output.slice_mut(s![b, .., ..]).assign(&result);
        }

        output
    }

    fn backward(&mut self, grad_output: &Self::Output) -> Result<Self::Input, ModelError> {
        // Get the cached input
        let input = self.last_input.read_ref()?;

        let (batch_size, seq_len, dim_in) = input.dim();
        let dim_out = self.weights.dim().1;

        let mut weight_grad_mut = self.weight_grad.mut_ref();
        let mut bias_grad_mut = self.bias_grad.mut_ref();

        // Lazy initialization of gradients on first backward pass
        if weight_grad_mut.is_none() {
            *weight_grad_mut = Some(Array2::zeros(self.weights.dim()));
        }
        if self.bias.is_some() && bias_grad_mut.is_none() {
            *bias_grad_mut = Some(Array1::zeros(dim_out));
        }

        // Parallelize gradient computation across batches
        #[allow(clippy::type_complexity)]
        let grad_results: Vec<(Array2<f32>, Option<Array1<f32>>, Array2<f32>)> = (0..batch_size)
            .into_par_iter()
            .map(|b| {
                let batch_input = input.slice(s![b, .., ..]);
                let batch_grad_output = grad_output.slice(s![b, .., ..]);

                // Gradient w.r.t. weights for this batch: input^T @ grad_output
                let weight_grad_batch = batch_input.t().dot(&batch_grad_output);

                // Gradient w.r.t. bias for this batch: sum over sequence dimension
                let bias_grad_batch = if self.bias.is_some() {
                    Some(batch_grad_output.sum_axis(ndarray::Axis(0)))
                } else {
                    None
                };

                // Gradient w.r.t. input: grad_output @ weights^T
                let grad_input_batch = batch_grad_output.dot(&self.weights.t());

                (weight_grad_batch, bias_grad_batch, grad_input_batch)
            })
            .collect();

        // Accumulate weight and bias gradients from all batches
        for (weight_grad_batch, bias_grad_batch, _) in &grad_results {
            if let Some(wg) = weight_grad_mut.as_mut() {
                *wg = &*wg + weight_grad_batch;
            }
            if let (Some(bg), Some(bias_grad)) = (bias_grad_mut.as_mut(), bias_grad_batch) {
                *bg = &*bg + bias_grad;
            }
        }

        // Combine input gradients back into 3D array
        let mut grad_input = Array3::<f32>::zeros((batch_size, seq_len, dim_in));
        for (b, (_, _, grad_input_batch)) in grad_results.into_iter().enumerate() {
            grad_input
                .slice_mut(s![b, .., ..])
                .assign(&grad_input_batch);
        }

        Ok(grad_input)
    }

    fn set_train(&mut self, _metrics_handle: TrainingMetricsHandle) {
        self.training = true;
    }

    fn set_eval(&mut self) {
        self.training = false;
    }

    fn get_params(&mut self) -> Vec<adamw::ParamHandle<'_>> {
        let mut params = vec![adamw::ParamHandle::Array2 {
            key: self.weight_grad.key(),
            data: &mut self.weights,
            grad: &self.weight_grad,
        }];

        // Add bias parameter if it exists
        if let Some(ref mut bias) = self.bias {
            params.push(adamw::ParamHandle::Array1 {
                key: self.bias_grad.key(),
                data: bias,
                grad: &self.bias_grad,
            });
        }

        params
    }
}

impl ZeroGrad for LinearLayer {
    fn zero_grad(&mut self) {
        if let Some(wg) = self.weight_grad.mut_ref().as_mut() {
            wg.fill(0.0);
        }
        if let Some(bg) = self.bias_grad.mut_ref().as_mut() {
            bg.fill(0.0);
        }
    }
}
