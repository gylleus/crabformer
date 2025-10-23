use crate::{
    errors::ModelError,
    layers::{Layer, LayerCacheParam, ZeroGrad, xavier_initialized_array},
    params::get_rng,
};
use ndarray::{Axis, s};

use ndarray::{Array2, Array3};

pub struct EmbeddingLayer {
    pub weights: Array2<f32>,
    pub gradients: Array2<f32>,
    // Cache input tokens for backward pass
    last_input_tokens: LayerCacheParam<Array2<u32>>,
    training: bool,
}

impl EmbeddingLayer {
    pub fn new(vocab_size: usize, embed_dim: usize, seed: Option<u64>) -> Self {
        let mut rng = get_rng(seed);

        let weights = xavier_initialized_array(vocab_size, embed_dim, &mut rng);

        let gradients = Array2::<f32>::zeros((vocab_size, embed_dim));

        Self {
            weights,
            gradients,
            last_input_tokens: LayerCacheParam::new("EmbeddingLayer::last_input_tokens"),
            training: false,
        }
    }

    /// Forward pass producing a 3D tensor of shape [batch_size, seq_length, embed_dim]
    // pub fn forward(&self, input_tokens: &Array2<u32>) -> Array3<f32> {
    //     let (batch_size, seq_length) = input_tokens.dim();
    //     let mut output = Array3::<f32>::zeros((batch_size, seq_length, self.weights.ncols()));
    //     self.forward_fill(input_tokens, &mut output);
    //     output
    // }

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

    // pub fn forward(&self, input_tokens: &Array2<u32>) -> Array3<f32> {
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

        // Scatter gradients to the embedding table
        // For each token, accumulate the gradient to its corresponding embedding row
        for batch_idx in 0..batch_size {
            for seq_idx in 0..seq_length {
                let token = input_tokens[[batch_idx, seq_idx]] as usize;
                let grad_slice = grad_output.slice(s![batch_idx, seq_idx, ..]);

                // Accumulate gradient for this token's embedding
                for (i, &grad_val) in grad_slice.iter().enumerate() {
                    self.gradients[[token, i]] += grad_val;
                }
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
}

impl ZeroGrad for EmbeddingLayer {
    fn zero_grad(&mut self) {
        self.gradients.fill(0.0);
    }
}
