use std::{
    cell::Cell,
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use ndarray::{Array2, Array3, Array4, Axis, s};
use rand::{SeedableRng, rngs::StdRng};

use crate::{
    errors::ModelError,
    layers::{
        Layer, LayerCacheParam, ZeroGrad, dropout::Dropout, linear::LinearLayer,
        normalization::Softmax,
    },
    params::MutableRng,
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

    // Training mode flag
    training: bool,

    // Cache for backward pass (using LayerCacheParam for interior mutability in forward)
    last_input: LayerCacheParam<Array3<f32>>,
    last_queries: LayerCacheParam<Array4<f32>>,
    last_keys: LayerCacheParam<Array4<f32>>,
    last_values: LayerCacheParam<Array4<f32>>,
    last_attention_weights: LayerCacheParam<Array4<f32>>,
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
            training: false,
            last_attention_weights: LayerCacheParam::new(
                "MultiHeadAttentionLayer::last_attention_weights",
            ),
            last_values: LayerCacheParam::new("MultiHeadAttentionLayer::last_values"),
            last_keys: LayerCacheParam::new("MultiHeadAttentionLayer::last_keys"),
            last_queries: LayerCacheParam::new("MultiHeadAttentionLayer::last_queries"),
            last_input: LayerCacheParam::new("MultiHeadAttentionLayer::last_input"),
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
}

impl Layer for MultiHeadAttentionLayer {
    type Input = Array3<f32>;
    type Output = Array3<f32>;

    fn forward(&self, input: &Self::Input) -> Self::Output {
        let (batch_size, seq_len, _embed_dim) = input.dim();

        // Compute queries, keys, and values (now handles 3D automatically)
        let queries_3d = self.query_weights.forward(input);
        let keys_3d = self.key_weights.forward(input);
        let values_3d = self.value_weights.forward(input);

        // Size of the dimension per head
        let head_dim = self.dim_out / self.num_heads;

        // Split the queries, keys, and values into multiple heads.
        // This uses the last axis to split into 2 more axes: num_heads and head_dim.
        let shape_4d = (batch_size, seq_len, self.num_heads, head_dim);

        let queries = queries_3d.to_shape(shape_4d).unwrap();
        let keys = keys_3d.to_shape(shape_4d).unwrap();
        let values = values_3d.to_shape(shape_4d).unwrap();

        // Rearrange the axes to bring num_heads upfront for easier processing.
        // (batch_size, num_heads, seq_len, head_dim)
        let permutated_axes = [0, 2, 1, 3];
        let queries = queries.permuted_axes(permutated_axes);
        let keys = keys.permuted_axes(permutated_axes);
        let values = values.permuted_axes(permutated_axes);

        // Cache Q, K, V if in training mode
        if self.training {
            *self.last_input.mut_ref() = Some(input.clone());
            *self.last_queries.mut_ref() = Some(queries.to_owned());
            *self.last_keys.mut_ref() = Some(keys.to_owned());
            *self.last_values.mut_ref() = Some(values.to_owned());
            *self.last_attention_weights.mut_ref() = Some(Array4::zeros((
                batch_size,
                self.num_heads,
                seq_len,
                seq_len,
            )));
        }

        let mut output = Array3::<f32>::zeros((batch_size, seq_len, self.dim_out));
        let mut attention_weights_all = if self.training {
            Some(Array4::<f32>::zeros((
                batch_size,
                self.num_heads,
                seq_len,
                seq_len,
            )))
        } else {
            None
        };

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
                attention_weights.softmax(0, None);

                // Apply dropout to attention weights (training only)
                attention_weights.apply_dropout(self.dropout_rate, &mut rng);

                if self.training {
                    attention_weights_all
                        .as_mut()
                        // Safe unwrap since we are in training mode
                        .unwrap()
                        .slice_mut(s![batch, head, .., ..])
                        .assign(&attention_weights);
                }

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

    fn backward(&mut self, grad_output: &Self::Output) -> Result<Self::Input, ModelError> {
        // Get cached values
        // let input = self
        //     .last_input
        //     .read_ref()
        //     .as_ref()
        //     .ok_or_else(|| ModelError::EmptyCache("MultiHeadAttentionLayer"))?
        //     .clone();
        // let queries = self
        //     .last_queries
        //     .read_ref()
        //     .as_ref()
        //     .ok_or_else(|| {
        //         ModelError::TrainingError("train() must be called before backward".into())
        //     })?
        //     .clone();
        // let keys = self
        //     .last_keys
        //     .read_ref()
        //     .as_ref()
        //     .ok_or_else(|| {
        //         ModelError::TrainingError("train() must be called before backward".into())
        //     })?
        //     .clone();
        // let values = self
        //     .last_values
        //     .read_ref()
        //     .as_ref()
        //     .ok_or_else(|| {
        //         ModelError::TrainingError("train() must be called before backward".into())
        //     })?
        //     .clone();

        let input = self.last_input.read_ref()?;
        let queries = self.last_queries.read_ref()?;
        let keys = self.last_keys.read_ref()?;
        let values = self.last_values.read_ref()?;
        let attention_weights = self.last_attention_weights.read_ref()?;

        let (batch_size, seq_len, _) = input.dim();
        let head_dim = self.dim_out / self.num_heads;

        // Initialize gradients for Q, K, V
        let mut grad_queries = Array4::<f32>::zeros(queries.dim());
        let mut grad_keys = Array4::<f32>::zeros(keys.dim());
        let mut grad_values = Array4::<f32>::zeros(values.dim());

        // Backprop through each head
        for batch in 0..batch_size {
            for head in 0..self.num_heads {
                // Get gradient for this head's output
                let start_idx = head * head_dim;
                let end_idx = start_idx + head_dim;
                let grad_head_output = grad_output.slice(s![batch, .., start_idx..end_idx]);

                // Cached values for this head
                let attn_weights = attention_weights.slice(s![batch, head, .., ..]);
                let head_values = values.slice(s![batch, head, .., ..]);
                let head_queries = queries.slice(s![batch, head, .., ..]);
                let head_keys = keys.slice(s![batch, head, .., ..]);

                // Gradient w.r.t. attention_weights @ V
                // grad_attn_weights = grad_output @ V^T
                let grad_attn_weights = grad_head_output.dot(&head_values.t());

                // Gradient w.r.t. V: attn_weights^T @ grad_output
                let grad_v = attn_weights.t().dot(&grad_head_output);
                grad_values
                    .slice_mut(s![batch, head, .., ..])
                    .assign(&grad_v);

                // Backprop through dropout (approximation: just pass through)
                let mut grad_attn_after_dropout = grad_attn_weights.clone();

                // Backprop through softmax
                // For softmax: dy/dx = softmax(x) * (grad - sum(grad * softmax(x)))
                let sum_grad = (&grad_attn_after_dropout * &attn_weights).sum();
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        grad_attn_after_dropout[[i, j]] =
                            attn_weights[[i, j]] * (grad_attn_after_dropout[[i, j]] - sum_grad);
                    }
                }

                // Backprop through scaling
                let dk = head_dim as f32;
                let grad_attn_scores = grad_attn_after_dropout.mapv(|x| x / dk.sqrt());

                // Backprop through Q @ K^T
                // grad_Q = grad_scores @ K
                let grad_q = grad_attn_scores.dot(&head_keys);
                grad_queries
                    .slice_mut(s![batch, head, .., ..])
                    .assign(&grad_q);

                // grad_K = grad_scores^T @ Q
                let grad_k = grad_attn_scores.t().dot(&head_queries);
                grad_keys.slice_mut(s![batch, head, .., ..]).assign(&grad_k);
            }
        }

        // Reshape gradients back: (batch, num_heads, seq, head_dim) -> (batch, seq, dim_out)
        let permute_back = [0, 2, 1, 3];
        let grad_queries_reshaped = grad_queries
            .permuted_axes(permute_back)
            .to_shape((batch_size, seq_len, self.dim_out))
            .unwrap()
            .to_owned();
        let grad_keys_reshaped = grad_keys
            .permuted_axes(permute_back)
            .to_shape((batch_size, seq_len, self.dim_out))
            .unwrap()
            .to_owned();
        let grad_values_reshaped = grad_values
            .permuted_axes(permute_back)
            .to_shape((batch_size, seq_len, self.dim_out))
            .unwrap()
            .to_owned();

        // Backprop through Q, K, V projections
        let grad_input_q = self.query_weights.backward(&grad_queries_reshaped)?;
        let grad_input_k = self.key_weights.backward(&grad_keys_reshaped)?;
        let grad_input_v = self.value_weights.backward(&grad_values_reshaped)?;

        // Sum gradients from all three paths
        let grad_input = grad_input_q + grad_input_k + grad_input_v;

        Ok(grad_input)
    }

    fn set_train(&mut self) {
        self.training = true;
        // Set training mode for linear layers
        self.query_weights.set_train();
        self.key_weights.set_train();
        self.value_weights.set_train();
    }

    fn set_eval(&mut self) {
        self.training = false;
        // Set eval mode for linear layers
        self.query_weights.set_eval();
        self.key_weights.set_eval();
        self.value_weights.set_eval();
        // Free cache memory
        *self.last_attention_weights.mut_ref() = None;
        *self.last_values.mut_ref() = None;
        *self.last_keys.mut_ref() = None;
        *self.last_queries.mut_ref() = None;
        *self.last_input.mut_ref() = None;
    }

    fn get_params(&mut self) -> Vec<crate::layers::ParamHandle> {
        let mut params = Vec::new();

        // Collect parameters from Q, K, V projection layers
        params.extend(self.query_weights.get_params());
        params.extend(self.key_weights.get_params());
        params.extend(self.value_weights.get_params());

        params
    }
}

impl ZeroGrad for MultiHeadAttentionLayer {
    fn zero_grad(&mut self) {
        self.query_weights.zero_grad();
        self.key_weights.zero_grad();
        self.value_weights.zero_grad();
    }
}
