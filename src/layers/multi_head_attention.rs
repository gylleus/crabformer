use std::{
    cell::Cell,
    sync::atomic::{AtomicU64, Ordering},
};

use ndarray::{Array2, Array3, Axis, s};
use rand::{SeedableRng, rngs::StdRng};

use crate::{
    errors::ModelError,
    layers::{
        Layer, dropout::Dropout, linear::LinearLayer, normalization::SoftMax,
        xavier_initialized_array,
    },
    params::{MutableRng, get_rng},
};

pub struct MultiHeadAttentionLayer {
    pub query_weights: LinearLayer,
    pub key_weights: LinearLayer,
    pub value_weights: LinearLayer,

    use_casual_mask: bool,
    dim_out: usize,
    num_heads: usize,
    dropout_rate: f32,
    // Internal counter to increment the RNG state for each forward pass while keeping reproducibility.
    rng: MutableRng,
}

// Multi-head attention layer block.
/// Has the same goal as the SelfAttentionLayer but splits the queries, keys, and values into multiple heads to allow the model to jointly attend to information from different representation subspaces at different positions.
/// This allows each head to learn different attention patterns that work together to give the model a richer context understanding of the input sequence.
impl MultiHeadAttentionLayer {
    pub fn new(
        dim_in: usize,
        dim_out: usize,
        num_heads: usize,
        dropout_rate: f32,
        seed: Option<u64>,
    ) -> Result<Self, ModelError> {
        if dim_out % num_heads != 0 {
            return Err(ModelError::DimensionMismatch(format!(
                "dim_out {} must be divisible by num_heads {}",
                dim_out, num_heads
            )));
        }

        let (query_weights, key_weights, value_weights) = (
            LinearLayer::new(dim_in, dim_out, seed),
            LinearLayer::new(dim_in, dim_out, seed),
            LinearLayer::new(dim_in, dim_out, seed),
        );

        Ok(Self {
            query_weights,
            key_weights,
            dim_out,
            num_heads,
            value_weights,
            use_casual_mask: false,
            dropout_rate,
            rng: MutableRng::new(seed),
        })
    }

    pub fn with_casual_mask(self) -> Self {
        Self {
            use_casual_mask: true,
            ..self
        }
    }

    pub fn with_qkv_bias(self) -> Self {
        Self {
            query_weights: self.query_weights.with_bias(),
            key_weights: self.key_weights.with_bias(),
            value_weights: self.value_weights.with_bias(),
            ..self
        }
    }

    fn causal_mask(&self, shape: (usize, usize)) -> Array2<f32> {
        // TODO: Pre-compute based on max sequence length and return a view based on input length.
        Array2::from_shape_fn(shape, |(i, j)| if j <= i { 0.0 } else { f32::NEG_INFINITY })
    }

    // pub fn forward(&self, input: &Array3<f32>) -> Array3<f32> {
    //     let (batch_size, seq_len, embed_dim) = input.dim();

    //     // Reshape input to 2D for matrix multiplication
    //     let input_2d = input.to_shape((batch_size * seq_len, embed_dim)).unwrap();

    //     // Compute queries, keys, and values with 2D dot products
    //     let queries_2d = self.query_weights.forward(&input_2d);
    //     let keys_2d = self.key_weights.forward(&input_2d);
    //     let values_2d = self.value_weights.forward(&input_2d);

    //     // Size of the dimension per head
    //     let head_dim = self.dim_out / self.num_heads;

    //     // Split the queries, keys, and values into multiple heads.
    //     // This uses the last axis to split into 2 more axes: num_heads and head_dim.
    //     let shape_4d = (batch_size, seq_len, self.num_heads, head_dim);

    //     let queries = queries_2d.to_shape(shape_4d).unwrap();
    //     let keys = keys_2d.to_shape(shape_4d).unwrap();
    //     let values = values_2d.to_shape(shape_4d).unwrap();

    //     // Rearrange the axes to bring num_heads upfront for easier processing.
    //     // (batch_size, num_heads, seq_len, head_dim)
    //     let permutated_axes = [0, 2, 1, 3];
    //     let queries = queries.permuted_axes(permutated_axes);
    //     let keys = keys.permuted_axes(permutated_axes);
    //     let values = values.permuted_axes(permutated_axes);

    //     let mut output = Array3::<f32>::zeros((batch_size, seq_len, self.dim_out));

    //     let seed = self.seed_counter.as_ref().map(|c| {
    //         // Seed increment is not thread-safe, but it's good enough for our purposes.
    //         let current = c.get();
    //         c.set(current.wrapping_add(1));
    //         current
    //     });
    //     let mut rng = get_rng(seed);

    //     // TODO: Optimize by using linalg libraries for efficient batched matrix multiplications.
    //     for batch in 0..batch_size {
    //         for head in 0..self.num_heads {
    //             // // Get the queries, keys, and values for the current batch and head
    //             let head_queries = queries.slice(s![batch, head, .., ..]); // (seq_len, head_dim)
    //             let head_keys = keys.slice(s![batch, head, .., ..]); // (seq_len, head_dim)
    //             let head_values = values.slice(s![batch, head, .., ..]); // (seq_len, head_dim)

    //             let attention_scores = head_queries.dot(&head_keys.t());

    //             let attention_scores = if self.use_casual_mask {
    //                 attention_scores + self.causal_mask((seq_len, seq_len))
    //             } else {
    //                 attention_scores
    //             };

    //             // Scale by sqrt(dk) to prevent large dot product values that can lead to vanishing gradients
    //             let dk = head_keys.ncols() as f32;
    //             let mut attention_weights = attention_scores.mapv(|x| x / dk.sqrt());

    //             // Normalize weights with softmax
    //             attention_weights.softmax(0);

    //             // Apply dropout to attention weights (training only)
    //             attention_weights.with_dropout(0.5, &mut rng);

    //             let context_vectors = attention_weights.dot(&head_values);

    //             // Determine the slice indices for this head in the output.
    //             // This is needed since we will go back from 4D to 3D by concatenating the head outputs.
    //             let start_idx = head * head_dim;
    //             let end_idx = start_idx + head_dim;

    //             // Add this head's context vectors to this batch's output
    //             output
    //                 .slice_mut(s![batch, .., start_idx..end_idx])
    //                 .assign(&context_vectors);
    //         }
    //     }

    //     output
    // }
}

impl Layer for MultiHeadAttentionLayer {
    /// input shape: (batch_size, seq_len, embed_dim)
    ///
    /// output shape: (batch_size, seq_len, embed_dim)
    fn forward(&self, input: &Array3<f32>) -> Array3<f32> {
        let (batch_size, seq_len, embed_dim) = input.dim();

        // Reshape input to 2D for matrix multiplication
        let input_2d = input.to_shape((batch_size * seq_len, embed_dim)).unwrap();

        // Compute queries, keys, and values with 2D dot products
        let queries_2d = self.query_weights.forward_2d(&input_2d);
        let keys_2d = self.key_weights.forward_2d(&input_2d);
        let values_2d = self.value_weights.forward_2d(&input_2d);

        // Size of the dimension per head
        let head_dim = self.dim_out / self.num_heads;

        // Split the queries, keys, and values into multiple heads.
        // This uses the last axis to split into 2 more axes: num_heads and head_dim.
        let shape_4d = (batch_size, seq_len, self.num_heads, head_dim);

        let queries = queries_2d.to_shape(shape_4d).unwrap();
        let keys = keys_2d.to_shape(shape_4d).unwrap();
        let values = values_2d.to_shape(shape_4d).unwrap();

        // Rearrange the axes to bring num_heads upfront for easier processing.
        // (batch_size, num_heads, seq_len, head_dim)
        let permutated_axes = [0, 2, 1, 3];
        let queries = queries.permuted_axes(permutated_axes);
        let keys = keys.permuted_axes(permutated_axes);
        let values = values.permuted_axes(permutated_axes);

        let mut output = Array3::<f32>::zeros((batch_size, seq_len, self.dim_out));

        let mut rng = self.rng.get_rng();

        // TODO: Optimize by using linalg libraries for efficient batched matrix multiplications.
        for batch in 0..batch_size {
            for head in 0..self.num_heads {
                // // Get the queries, keys, and values for the current batch and head
                let head_queries = queries.slice(s![batch, head, .., ..]); // (seq_len, head_dim)
                let head_keys = keys.slice(s![batch, head, .., ..]); // (seq_len, head_dim)
                let head_values = values.slice(s![batch, head, .., ..]); // (seq_len, head_dim)

                let attention_scores = head_queries.dot(&head_keys.t());

                let attention_scores = if self.use_casual_mask {
                    attention_scores + self.causal_mask((seq_len, seq_len))
                } else {
                    attention_scores
                };

                // Scale by sqrt(dk) to prevent large dot product values that can lead to vanishing gradients
                let dk = head_keys.ncols() as f32;
                let mut attention_weights = attention_scores.mapv(|x| x / dk.sqrt());

                // Normalize weights with softmax
                attention_weights.softmax(0);

                // Apply dropout to attention weights (training only)
                attention_weights.apply_dropout(self.dropout_rate, &mut rng);

                let context_vectors = attention_weights.dot(&head_values);

                // Determine the slice indices for this head in the output.
                // This is needed since we will go back from 4D to 3D by concatenating the head outputs.
                let start_idx = head * head_dim;
                let end_idx = start_idx + head_dim;

                // Add this head's context vectors to this batch's output
                output
                    .slice_mut(s![batch, .., start_idx..end_idx])
                    .assign(&context_vectors);
            }
        }

        output
    }
}
