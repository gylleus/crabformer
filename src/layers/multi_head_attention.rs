use std::{
    cell::Cell,
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use ndarray::{Array2, Array3, Array4, s};
use rand::{SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    errors::ModelError,
    layers::{
        Layer, LayerCacheParam, ZeroGrad, dropout::Dropout, linear::LinearLayer,
        normalization::Softmax,
    },
    metrics::TrainingMetricsHandle,
    params::SEQUENCE_LENGTH,
};

#[derive(Serialize, Deserialize)]
pub struct MultiHeadAttentionLayer {
    pub query_weights: LinearLayer,
    pub key_weights: LinearLayer,
    pub value_weights: LinearLayer,

    use_casual_mask: bool,
    dim_out: usize,
    num_heads: usize,
    dropout_rate: f32,

    // Training mode flag
    training: bool,

    // Locally cached causal mask (lazily initialized during first forward pass)
    #[serde(skip)]
    causal_mask: LayerCacheParam<Array2<f32>>,

    // Cache for backward pass (using LayerCacheParam for interior mutability in forward)
    #[serde(skip)]
    last_input: LayerCacheParam<Array3<f32>>,
    #[serde(skip)]
    last_queries: LayerCacheParam<Array4<f32>>,
    #[serde(skip)]
    last_keys: LayerCacheParam<Array4<f32>>,
    #[serde(skip)]
    last_values: LayerCacheParam<Array4<f32>>,
    #[serde(skip)]
    last_attention_weights: LayerCacheParam<Array4<f32>>,
    #[serde(skip)]
    attention_dropout_mask: LayerCacheParam<Vec<bool>>,

    name: String,
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
        name: Option<String>,
    ) -> Result<Self, ModelError> {
        let name = name.unwrap_or("MultiHeadAttentionLayer".into());

        if dim_out % num_heads != 0 {
            return Err(ModelError::DimensionMismatch(format!(
                "dim_out {} must be divisible by num_heads {}",
                dim_out, num_heads
            )));
        }

        let (query_weights, key_weights, value_weights) = (
            LinearLayer::new(dim_in, dim_out, Some(format!("{}::query_weights", name))),
            LinearLayer::new(dim_in, dim_out, Some(format!("{}::key_weights", name))),
            LinearLayer::new(dim_in, dim_out, Some(format!("{}::value_weights", name))),
        );

        Ok(Self {
            query_weights,
            key_weights,
            dim_out,
            num_heads,
            value_weights,
            use_casual_mask: false,
            dropout_rate,
            training: false,
            causal_mask: LayerCacheParam::new(format!("{}::cached_causal_mask", name)),
            last_attention_weights: LayerCacheParam::new(format!(
                "{}::last_attention_weights",
                name
            )),
            last_values: LayerCacheParam::new(format!("{}::last_values", name)),
            last_keys: LayerCacheParam::new(format!("{}::last_keys", name)),
            last_queries: LayerCacheParam::new(format!("{}::last_queries", name)),
            last_input: LayerCacheParam::new(format!("{}::last_input", name)),
            attention_dropout_mask: LayerCacheParam::new(format!("{}::attention_dropout_mask", name)),
            name,
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

    /// Generates (or retrieves cached) causal mask for given sequence length.
    /// This mask will be added to the attention scores to prevent attending to future tokens.
    fn get_causal_mask(&self, seq_len: usize) -> Array2<f32> {
        let mut cache = self.causal_mask.mut_ref();

        // Lazily initialize the mask on first use
        if cache.is_none() {
            *cache = Some(Array2::from_shape_fn(
                (SEQUENCE_LENGTH, SEQUENCE_LENGTH),
                |(i, j)| {
                    // Allowed positions remain unchanged, blocked future positions get -inf
                    if j <= i { 0.0 } else { f32::NEG_INFINITY }
                },
            ));
        }

        // Return a slice of the cached mask for the current sequence length
        cache
            .as_ref()
            .unwrap()
            .slice(s![..seq_len, ..seq_len])
            .to_owned()
    }
}

impl Layer for MultiHeadAttentionLayer {
    type Input = Array3<f32>;
    type Output = Array3<f32>;

    fn name(&self) -> &str {
        &self.name
    }

    fn forward(&self, input: &Self::Input) -> Self::Output {
        let (batch_size, seq_len, _embed_dim) = input.dim();

        // Compute queries, keys, and values using linear projections
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

        // Batched attention computation - much faster than nested loops
        // Reshape to (batch*heads, seq, head_dim) for batched matmul
        let batch_heads = batch_size * self.num_heads;
        let batch_heads_dim_3d = (batch_heads, seq_len, head_dim);

        let queries_reshaped = queries.to_shape(batch_heads_dim_3d).unwrap();
        let keys_reshaped = keys.to_shape(batch_heads_dim_3d).unwrap();
        let values_reshaped = values.to_shape(batch_heads_dim_3d).unwrap();

        // Compute attention scores for all batches and heads at once (parallelized)
        // Q @ K^T for each of the batch*head matrices
        let attention_scores_vec: Vec<Array2<f32>> = (0..batch_heads)
            .into_par_iter()
            .map(|i| {
                let queries_i = queries_reshaped.slice(s![i, .., ..]);
                let keys_i = keys_reshaped.slice(s![i, .., ..]);
                queries_i.dot(&keys_i.t())
            })
            .collect();

        let mut attention_scores = Array3::<f32>::zeros((batch_heads, seq_len, seq_len));
        for (i, scores) in attention_scores_vec.into_iter().enumerate() {
            attention_scores.slice_mut(s![i, .., ..]).assign(&scores);
        }

        // Apply causal mask if needed
        if self.use_casual_mask {
            let mask = self.get_causal_mask(seq_len);

            // Apply the mask to each batch_head attention score matrix
            for i in 0..batch_heads {
                let mut current = attention_scores.slice_mut(s![i, .., ..]);

                current += &mask;
            }
        }

        // Scale by sqrt(dk) for stability
        let dk = head_dim as f32;
        attention_scores.mapv_inplace(|x| x / dk.sqrt());

        // Apply softmax to each (seq_len, seq_len) matrix (parallelized)
        let softmax_results: Vec<Array2<f32>> = (0..batch_heads)
            .into_par_iter()
            .map(|i| {
                let mut attn_matrix = attention_scores.slice(s![i, .., ..]).to_owned();
                attn_matrix.softmax(0, None);
                attn_matrix
            })
            .collect();

        for (i, result) in softmax_results.into_iter().enumerate() {
            attention_scores.slice_mut(s![i, .., ..]).assign(&result);
        }

        // Cache attention weights BEFORE dropout for backward pass
        if self.training {
            let attention_weights_all = attention_scores
                .to_shape((batch_size, self.num_heads, seq_len, seq_len))
                .unwrap()
                .to_owned();
            *self.last_attention_weights.mut_ref() = Some(attention_weights_all);
        }

        // Apply dropout (training only)
        if self.training {
            let mask = attention_scores.apply_dropout(self.dropout_rate);
            *self.attention_dropout_mask.mut_ref() = mask;
        }

        // Compute context vectors: attention_weights @ V (parallelized)
        let context_vec: Vec<Array2<f32>> = (0..batch_heads)
            .into_par_iter()
            .map(|i| {
                let attn = attention_scores.slice(s![i, .., ..]);
                let v = values_reshaped.slice(s![i, .., ..]);
                attn.dot(&v)
            })
            .collect();

        let mut context = Array3::<f32>::zeros((batch_heads, seq_len, head_dim));
        for (i, ctx) in context_vec.into_iter().enumerate() {
            context.slice_mut(s![i, .., ..]).assign(&ctx);
        }

        // Reshape back to (batch, heads, seq, head_dim) then (batch, seq, heads, head_dim)
        let context_4d = context
            .to_shape((batch_size, self.num_heads, seq_len, head_dim))
            .unwrap();
        let context_reordered = context_4d.permuted_axes([0, 2, 1, 3]);

        // Reshape to (batch, seq, dim_out) to concatenate heads
        let output = context_reordered
            .to_shape((batch_size, seq_len, self.dim_out))
            .unwrap()
            .to_owned();

        output
    }

    fn backward(&mut self, grad_output: &Self::Output) -> Result<Self::Input, ModelError> {
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

        // Get dropout mask if it exists (clone for parallel access)
        let dropout_mask_opt = self.attention_dropout_mask.mut_ref().clone();

        // Backprop through each head (parallelized across batch*heads)
        let batch_heads = batch_size * self.num_heads;
        let dropout_rate = self.dropout_rate;
        let keep_prob = 1.0 - dropout_rate;

        let grad_results: Vec<(usize, usize, Array2<f32>, Array2<f32>, Array2<f32>)> = (0..batch_heads)
            .into_par_iter()
            .map(|idx| {
                let batch = idx / self.num_heads;
                let head = idx % self.num_heads;

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
                let grad_attn_weights = grad_head_output.dot(&head_values.t());

                // Gradient w.r.t. V: attn_weights^T @ grad_output
                let grad_v = attn_weights.t().dot(&grad_head_output);

                // Backprop through dropout - apply the same mask used in forward pass
                let mut grad_attn_after_dropout = grad_attn_weights.clone();

                if let Some(ref mask) = dropout_mask_opt {
                    let offset = idx * seq_len * seq_len;
                    for i in 0..seq_len {
                        for j in 0..seq_len {
                            let mask_idx = offset + i * seq_len + j;
                            if !mask[mask_idx] {
                                grad_attn_after_dropout[[i, j]] = 0.0;
                            } else {
                                grad_attn_after_dropout[[i, j]] /= keep_prob;
                            }
                        }
                    }
                }

                // Backprop through softmax
                for i in 0..seq_len {
                    let mut sum_grad_row = 0.0;
                    for j in 0..seq_len {
                        sum_grad_row += grad_attn_after_dropout[[i, j]] * attn_weights[[i, j]];
                    }
                    for j in 0..seq_len {
                        grad_attn_after_dropout[[i, j]] =
                            attn_weights[[i, j]] * (grad_attn_after_dropout[[i, j]] - sum_grad_row);
                    }
                }

                // Backprop through scaling
                let dk = head_dim as f32;
                let grad_attn_scores = grad_attn_after_dropout.mapv(|x| x / dk.sqrt());

                // Backprop through Q @ K^T
                let grad_q = grad_attn_scores.dot(&head_keys);
                let grad_k = grad_attn_scores.t().dot(&head_queries);

                (batch, head, grad_q, grad_k, grad_v)
            })
            .collect();

        // Assign results back to gradient arrays
        for (batch, head, grad_q, grad_k, grad_v) in grad_results {
            grad_queries.slice_mut(s![batch, head, .., ..]).assign(&grad_q);
            grad_keys.slice_mut(s![batch, head, .., ..]).assign(&grad_k);
            grad_values.slice_mut(s![batch, head, .., ..]).assign(&grad_v);
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

    fn set_train(&mut self, metrics_handle: TrainingMetricsHandle) {
        self.training = true;

        // Set training mode for linear layers
        self.query_weights.set_train(metrics_handle.clone());
        self.key_weights.set_train(metrics_handle.clone());
        self.value_weights.set_train(metrics_handle.clone());
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
