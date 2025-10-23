use crate::{
    errors::ModelError,
    layers::{Layer, LayerCacheParam, ZeroGrad, xavier_initialized_array},
};
use ndarray::{Axis, s};

use ndarray::{Array2, Array3};

pub struct EmbeddingLayer {
    pub weights: Array2<f32>,
    // pub gradients: Array2<f32>,
    // Cache input tokens for backward pass
    weight_grad: LayerCacheParam<Array2<f32>>,
    last_input_tokens: LayerCacheParam<Array2<u32>>,
    training: bool,

    name: String,
}

impl EmbeddingLayer {
    pub fn new(vocab_size: usize, embed_dim: usize, name: Option<String>) -> Self {
        let name = name.unwrap_or("EmbeddingLayer".into());

        let weights = xavier_initialized_array(vocab_size, embed_dim);

        Self {
            weights,
            weight_grad: LayerCacheParam::new(format!("{}::weight_grad", name)),
            last_input_tokens: LayerCacheParam::new(format!("{}::last_input_tokens", name)),
            training: false,
            name,
        }
    }

    /// Fills a preallocated output array with the embeddings for the input tokens.
    pub fn forward_fill(&self, input_tokens: &Array2<u32>, output: &mut Array3<f32>) {
        let (batch_size, seq_length) = input_tokens.dim();
        assert_eq!(output.dim(), (batch_size, seq_length, self.weights.ncols()));
        // assert_eq!(output.dim(), (input_tokens.len(), self.weights.ncols()));

        for batch_idx in 0..batch_size {
            for seq_idx in 0..seq_length {
                let token = input_tokens[[batch_idx, seq_idx]];
                output
                    .slice_mut(s![batch_idx, seq_idx, ..])
                    .assign(&self.weights.row(token as usize));
            }
        }
    }
}

impl Layer for EmbeddingLayer {
    type Input = Array2<u32>;
    type Output = Array3<f32>;

    fn name(&self) -> &str {
        &self.name
    }

    fn forward(&self, input_tokens: &Self::Input) -> Self::Output {
        let (batch_size, seq_length) = input_tokens.dim();
        let mut output = Array3::<f32>::zeros((batch_size, seq_length, self.weights.ncols()));
        self.forward_fill(input_tokens, &mut output);

        if self.training {
            *self.last_input_tokens.mut_ref() = Some(input_tokens.clone());
        }

        output
    }

    fn backward(&mut self, grad_output: &Self::Output) -> Result<Self::Input, ModelError> {
        let input_tokens = self.last_input_tokens.read_ref()?;

        let (batch_size, seq_length, _embed_dim) = grad_output.dim();

        let mut weight_grad_mut = self.weight_grad.mut_ref();

        if weight_grad_mut.is_none() {
            *weight_grad_mut = Some(Array2::zeros(self.weights.dim()));
        }

        // Scatter gradients to the embedding table
        // For each token, accumulate the gradient to its corresponding embedding row
        let weight_grad = weight_grad_mut.as_mut().unwrap();

        for batch_idx in 0..batch_size {
            for seq_idx in 0..seq_length {
                let token = input_tokens[[batch_idx, seq_idx]] as usize;
                let grad_slice = grad_output.slice(s![batch_idx, seq_idx, ..]);

                // Accumulate gradient for this token's embedding using vectorized operation
                let mut row = weight_grad.row_mut(token);
                row += &grad_slice;
            }
        }
        Ok(input_tokens.clone())
    }

    fn set_train(&mut self) {
        self.training = true;
    }

    fn set_eval(&mut self) {
        self.training = false;
        self.last_input_tokens.clear();
    }

    fn get_params(&mut self) -> Vec<crate::adamw::ParamHandle> {
        vec![crate::adamw::ParamHandle::Array2 {
            key: self.weight_grad.id(),
            data: &mut self.weights,
            grad: &self.weight_grad,
        }]
    }
}

impl ZeroGrad for EmbeddingLayer {
    fn zero_grad(&mut self) {
        if let Some(wg) = self.weight_grad.mut_ref().as_mut() {
            wg.fill(0.0);
        }
    }
}
