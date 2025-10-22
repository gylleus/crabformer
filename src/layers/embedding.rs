use crate::{layers::xavier_initialized_array, params::get_rng};
use ndarray::{Axis, s};

use ndarray::{Array2, Array3};

pub struct EmbeddingLayer {
    pub weights: Array2<f32>,
    pub gradients: Array2<f32>,
}

impl EmbeddingLayer {
    pub fn new(vocab_size: usize, embed_dim: usize, seed: Option<u64>) -> Self {
        let mut rng = get_rng(seed);

        let weights = xavier_initialized_array(vocab_size, embed_dim, &mut rng);

        let gradients = Array2::<f32>::zeros((vocab_size, embed_dim));

        Self { weights, gradients }
    }

    /// Forward pass producing a 3D tensor of shape [batch_size, seq_length, embed_dim]
    pub fn forward(&self, input_tokens: &Array2<u32>) -> Array3<f32> {
        let (batch_size, seq_length) = input_tokens.dim();
        let mut output = Array3::<f32>::zeros((batch_size, seq_length, self.weights.ncols()));
        self.forward_fill(input_tokens, &mut output);
        output
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
