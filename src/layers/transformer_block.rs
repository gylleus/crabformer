use std::time::{Duration, Instant};

use ndarray::Array3;
use serde::{Deserialize, Serialize};

use crate::{
    adamw::ParamHandle,
    errors::ModelError,
    layers::{
        Layer, LayerCacheParam, ZeroGrad, dropout::Dropout, linear::FeedForwardLayer,
        multi_head_attention::MultiHeadAttentionLayer, normalization::LayerNormLayer,
    },
    metrics::TrainingMetricsHandle,
};

#[derive(Serialize, Deserialize)]
pub struct TransformerBlock {
    pub self_attention: MultiHeadAttentionLayer,
    pub feed_forward: FeedForwardLayer,
    pub layer_norm: LayerNormLayer,
    pub feed_forward_layer_norm: LayerNormLayer,
    pub dropout_rate: f32,

    // Training mode flag
    training: bool,

    name: String,

    #[serde(skip)]
    metrics_handle: Option<TrainingMetricsHandle>,

    // Cache dropout masks for backward pass
    #[serde(skip)]
    attention_dropout_mask: LayerCacheParam<Vec<bool>>,
    #[serde(skip)]
    ff_dropout_mask: LayerCacheParam<Vec<bool>>,
}

impl TransformerBlock {
    pub fn new(
        seq_length: usize,
        dim_model: usize,
        num_heads: usize,
        dim_ff: usize,
        dropout_rate: f32,
        qkv_bias: bool,
        name: Option<String>,
    ) -> Result<Self, ModelError> {
        let name = name.unwrap_or("TransformerBlock".into());

        let attention_layer = MultiHeadAttentionLayer::new(
            dim_model,
            dim_model,
            num_heads,
            seq_length,
            dropout_rate,
            Some(format!("{}::self_attention", name)),
        )?
        .with_casual_mask();

        let attention_layer = if qkv_bias {
            attention_layer.with_qkv_bias()
        } else {
            attention_layer
        };

        Ok(Self {
            self_attention: attention_layer,
            feed_forward: FeedForwardLayer::new(
                dim_model,
                dim_ff,
                Some(format!("{}::feed_forward", name)),
            ),
            layer_norm: LayerNormLayer::new(dim_model, Some(format!("{}::layer_norm", name))),
            feed_forward_layer_norm: LayerNormLayer::new(
                dim_model,
                Some(format!("{}::feed_forward_layer_norm", name)),
            ),
            dropout_rate,
            training: false,
            name: name.clone(),
            metrics_handle: None,
            attention_dropout_mask: LayerCacheParam::new(format!(
                "{}::attention_dropout_mask",
                name
            )),
            ff_dropout_mask: LayerCacheParam::new(format!("{}::ff_dropout_mask", name)),
        })
    }
}

impl Layer for TransformerBlock {
    type Input = Array3<f32>;
    type Output = Array3<f32>;

    fn name(&self) -> &str {
        &self.name
    }

    fn forward(&self, input: &Self::Input) -> Self::Output {
        let start_time = Instant::now();
        let mut total_norm_duration = Duration::ZERO;

        let norm_start = Instant::now();
        let layer_normalized_input = self.layer_norm.forward(input);
        total_norm_duration += norm_start.elapsed();

        // Self-attention sub-layer
        let attn_start = Instant::now();
        let mut attention_output = self.self_attention.forward(&layer_normalized_input);
        let attention_duration = attn_start.elapsed();

        // Apply dropout after attention (only during training)
        if self.training {
            let mask = attention_output.apply_dropout(self.dropout_rate);
            *self.attention_dropout_mask.mut_ref() = mask;
        }

        // Residual connection (skip connection)
        attention_output += input;

        // Normalize again before feed-forward
        let norm_start = Instant::now();
        let layer_normalized_attention = self.feed_forward_layer_norm.forward(&attention_output);
        total_norm_duration += norm_start.elapsed();

        // Feed-forward sub-layer
        let ff_start = Instant::now();
        let mut feed_forward_output = self.feed_forward.forward(&layer_normalized_attention);
        let ff_duration = ff_start.elapsed();

        // Apply dropout after feed-forward (only during training)
        if self.training {
            let mask = feed_forward_output.apply_dropout(self.dropout_rate);
            *self.ff_dropout_mask.mut_ref() = mask;
            if let Some(metrics_handle) = &self.metrics_handle {
                // metrics_handle
                //     .lock()
                //     .transformer_block_duration
                //     .forward_duration += start_time.elapsed();

                let mut metrics = metrics_handle.lock();
                metrics.attention_duration.add_forward(attention_duration);
                metrics.layer_norm_duration.add_forward(total_norm_duration);
                metrics.feed_forward_duration.add_forward(ff_duration);
                metrics
                    .transformer_block_duration
                    .add_forward(start_time.elapsed());
            }
        }

        // Second residual connection
        attention_output += &feed_forward_output;

        attention_output
    }

    fn backward(&mut self, grad_output: &Self::Output) -> Result<Self::Input, ModelError> {
        let start_time = Instant::now();
        let mut total_norm_duration = Duration::ZERO;

        // Start with gradient flowing back from output
        let grad = grad_output.clone();

        // Backprop through second residual connection
        // output = attention_output + feed_forward_output
        // So gradients flow to both paths
        let mut grad_ff_output = grad.clone();
        let grad_attention_after_residual = grad.clone();

        // Backprop through dropout for feed-forward
        if let Some(ref mask) = *self.ff_dropout_mask.mut_ref() {
            grad_ff_output.apply_dropout_backward(mask, self.dropout_rate);
        }

        // Backprop through feed-forward layer
        let ff_start = Instant::now();
        let grad_from_ff = self.feed_forward.backward(&grad_ff_output)?;
        let ff_duration = ff_start.elapsed();

        // Backprop through second layer norm
        let norm_start = Instant::now();
        let grad_attention_from_norm = self.feed_forward_layer_norm.backward(&grad_from_ff)?;
        total_norm_duration += norm_start.elapsed();

        // Combine gradients from both residual paths
        let mut grad_attention_total = &grad_attention_from_norm + &grad_attention_after_residual;

        // Backprop through dropout after attention
        if let Some(ref mask) = *self.attention_dropout_mask.mut_ref() {
            grad_attention_total.apply_dropout_backward(mask, self.dropout_rate);
        }

        // Backprop through self-attention
        let attention_start = Instant::now();
        let grad_layer_norm_input = self.self_attention.backward(&grad_attention_total)?;
        let attention_duration = attention_start.elapsed();

        // Backprop through first layer norm
        let norm_start = Instant::now();
        let grad_input_from_norm = self.layer_norm.backward(&grad_layer_norm_input)?;
        total_norm_duration += norm_start.elapsed();

        // Combine gradients from both residual paths
        // grad_input_from_norm: gradient through layer_norm and attention
        // grad_attention_after_residual: gradient through the skip connection
        let grad_input = grad_input_from_norm + &grad_attention_after_residual;

        // Update metrics
        if let Some(metrics_handle) = &self.metrics_handle {
            let mut metrics = metrics_handle.lock();

            metrics.attention_duration.add_backward(attention_duration);
            metrics
                .layer_norm_duration
                .add_backward(total_norm_duration);
            metrics.feed_forward_duration.add_backward(ff_duration);
            metrics
                .transformer_block_duration
                .add_backward(start_time.elapsed());
        }

        Ok(grad_input)
    }

    fn set_train(&mut self, metrics_handle: TrainingMetricsHandle) {
        self.training = true;
        self.metrics_handle = Some(metrics_handle.clone());

        // Set sublayers to training mode
        self.self_attention.set_train(metrics_handle.clone());
        self.feed_forward.set_train(metrics_handle.clone());
        self.layer_norm.set_train(metrics_handle.clone());
        self.feed_forward_layer_norm
            .set_train(metrics_handle.clone());
    }

    fn set_eval(&mut self) {
        self.training = false;

        // Set sublayers to eval mode
        self.self_attention.set_eval();
        self.feed_forward.set_eval();
        self.layer_norm.set_eval();
        self.feed_forward_layer_norm.set_eval();
    }

    fn get_params(&mut self) -> Vec<ParamHandle<'_>> {
        let mut params = Vec::new();

        // Collect parameters from all sub-layers
        params.extend(self.layer_norm.get_params());
        params.extend(self.feed_forward_layer_norm.get_params());
        params.extend(self.self_attention.get_params());
        params.extend(self.feed_forward.get_params());

        params
    }
}

impl ZeroGrad for TransformerBlock {
    fn zero_grad(&mut self) {
        self.self_attention.zero_grad();
        self.feed_forward.zero_grad();
        self.layer_norm.zero_grad();
        self.feed_forward_layer_norm.zero_grad();
    }
}
