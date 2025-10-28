use ndarray::{Array2, Array3, s};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    adamw,
    errors::ModelError,
    layers::{
        Layer, LayerCacheParam, ZeroGrad,
        activation::{Gelu, gelu_derivative},
        linear::LinearLayer,
    },
    metrics::TrainingMetricsHandle,
};

#[derive(Serialize, Deserialize)]
/// Feed-Forward Layer with two linear transformations and GELU activation
pub struct FeedForwardLayer {
    pub linear1: LinearLayer,
    pub linear2: LinearLayer,

    #[serde(skip)]
    // Cache for backward pass (stores hidden state before GELU activation)
    last_hidden: LayerCacheParam<Array3<f32>>,
    training: bool,

    name: String,
}

impl FeedForwardLayer {
    pub fn new(dim_model: usize, dim_ff: usize, name: Option<String>) -> Self {
        let name = name.unwrap_or("FeedForwardLayer".into());

        let linear1 = LinearLayer::new(dim_model, dim_ff, Some(format!("{}::linear1", name)));
        let linear2 = LinearLayer::new(dim_ff, dim_model, Some(format!("{}::linear2", name)));

        Self {
            linear1,
            linear2,
            last_hidden: LayerCacheParam::new(format!("{}::last_hidden", name)),
            training: false,
            name,
        }
    }
}

impl Layer for FeedForwardLayer {
    type Input = Array3<f32>;
    type Output = Array3<f32>;

    fn name(&self) -> &str {
        &self.name
    }

    fn forward(&self, input: &Self::Input) -> Self::Output {
        let hidden = self.linear1.forward(input);

        // Cache the hidden state before activation for backward pass
        if self.training {
            *self.last_hidden.mut_ref() = Some(hidden.clone());
        }

        let (batch_size, seq_len, dim_ff) = hidden.dim();

        // Apply GELU activation in parallel across batch dimension
        let activated_batches: Vec<Array2<f32>> = (0..batch_size)
            .into_par_iter()
            .map(|b| {
                let mut batch_hidden = hidden.slice(s![b, .., ..]).to_owned();
                batch_hidden.apply_gelu();
                batch_hidden
            })
            .collect();

        // Reconstruct the activated hidden state
        let mut activated_hidden = Array3::<f32>::zeros((batch_size, seq_len, dim_ff));
        for (b, batch) in activated_batches.into_iter().enumerate() {
            activated_hidden.slice_mut(s![b, .., ..]).assign(&batch);
        }

        self.linear2.forward(&activated_hidden)
    }

    fn backward(&mut self, grad_output: &Self::Output) -> Result<Self::Input, ModelError> {
        let hidden = self.last_hidden.read_ref()?;

        // Backprop through second linear layer
        let grad_hidden = self.linear2.backward(grad_output)?;

        let (batch_size, seq_len, dim_ff) = grad_hidden.dim();

        // Backprop through GELU activation in parallel across batch dimension
        // GELU'(x) needs the original input to GELU (which is the hidden state)
        let grad_hidden_batches: Vec<Array2<f32>> = (0..batch_size)
            .into_par_iter()
            .map(|b| {
                let batch_grad_hidden = grad_hidden.slice(s![b, .., ..]);
                let batch_hidden = hidden.slice(s![b, .., ..]);

                // Apply GELU derivative element-wise
                &batch_grad_hidden.to_owned() * &batch_hidden.mapv(gelu_derivative)
            })
            .collect();

        // Reconstruct the gradient after GELU backprop
        let mut grad_hidden_after_gelu = Array3::<f32>::zeros((batch_size, seq_len, dim_ff));
        for (b, batch) in grad_hidden_batches.into_iter().enumerate() {
            grad_hidden_after_gelu
                .slice_mut(s![b, .., ..])
                .assign(&batch);
        }

        // Backprop through first linear layer
        let grad_input = self.linear1.backward(&grad_hidden_after_gelu)?;

        Ok(grad_input)
    }

    fn set_train(&mut self, metrics_handle: TrainingMetricsHandle) {
        self.training = true;
        self.linear1.set_train(metrics_handle.clone());
        self.linear2.set_train(metrics_handle.clone());
    }

    fn set_eval(&mut self) {
        self.training = false;
        self.linear1.set_eval();
        self.linear2.set_eval();
    }

    fn get_params(&mut self) -> Vec<adamw::ParamHandle<'_>> {
        let mut params = Vec::new();

        // Collect parameters from both linear layers
        params.extend(self.linear1.get_params());
        params.extend(self.linear2.get_params());

        params
    }
}

impl ZeroGrad for FeedForwardLayer {
    fn zero_grad(&mut self) {
        self.linear1.zero_grad();
        self.linear2.zero_grad();
    }
}
