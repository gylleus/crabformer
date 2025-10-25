use std::fmt::format;

use ndarray::Array3;
use serde::{Deserialize, Serialize};

use crate::{
    errors::ModelError,
    layers::{
        Layer, LayerCacheParam, ZeroGrad, dropout::Dropout, linear::FeedForwardLayer,
        multi_head_attention::MultiHeadAttentionLayer, normalization::LayerNormLayer,
    },
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
}

impl TransformerBlock {
    pub fn new(
        dim_model: usize,
        num_heads: usize,
        dim_ff: usize,
        dropout_rate: f32,
        name: Option<String>,
    ) -> Result<Self, ModelError> {
        let name = name.unwrap_or("TransformerBlock".into());

        let attention_layer = MultiHeadAttentionLayer::new(
            dim_model,
            dim_model,
            num_heads,
            dropout_rate,
            Some(format!("{}::self_attention", name)),
        )?;

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
            name,
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
        let layer_normalized_input = self.layer_norm.forward(input);

        // Self-attention sub-layer
        let mut attention_output = self.self_attention.forward(&layer_normalized_input);
        // .to_array3()?;

        // Apply dropout after attention (only during training)
        if self.training {
            attention_output.apply_dropout(self.dropout_rate);
        }

        // Residual connection (skip connection)
        attention_output += input;

        // Normalize again before feed-forward
        let layer_normalized_attention = self.feed_forward_layer_norm.forward(&attention_output);

        // Feed-forward sub-layer
        let mut feed_forward_output = self.feed_forward.forward(&layer_normalized_attention);

        // Apply dropout after feed-forward (only during training)
        if self.training {
            feed_forward_output.apply_dropout(self.dropout_rate);
        }

        // Second residual connection
        attention_output += &feed_forward_output;

        attention_output
    }

    fn backward(&mut self, grad_output: &Self::Output) -> Result<Self::Input, ModelError> {
        // Start with gradient flowing back from output
        let grad = grad_output.clone();

        // Backprop through second residual connection
        // output = attention_output + feed_forward_output
        // So gradients flow to both paths
        let grad_ff_output = grad.clone();
        let grad_attention_after_residual = grad.clone();

        // Backprop through dropout (approximation: just pass through)
        // In training, dropout zeros out some values, but we'll approximate by passing through

        // Backprop through feed-forward layer
        let grad_from_ff = self.feed_forward.backward(&grad_ff_output)?;

        // Backprop through second layer norm
        let grad_attention_from_norm = self.feed_forward_layer_norm.backward(&grad_from_ff)?;

        // Combine gradients from both residual paths
        let grad_attention_total = &grad_attention_from_norm + &grad_attention_after_residual;

        // Backprop through dropout after attention (approximation: pass through)

        // Backprop through self-attention
        let grad_layer_norm_input = self.self_attention.backward(&grad_attention_total)?;

        // Backprop through first layer norm
        let grad_input_from_norm = self.layer_norm.backward(&grad_layer_norm_input)?;

        // Combine gradients from both residual paths
        let grad_input = grad_input_from_norm + &grad_attention_total;

        Ok(grad_input)
    }

    fn set_train(&mut self) {
        self.training = true;

        // Set sublayers to training mode
        self.self_attention.set_train();
        self.feed_forward.set_train();
        self.layer_norm.set_train();
        self.feed_forward_layer_norm.set_train();
    }

    fn set_eval(&mut self) {
        self.training = false;

        // Set sublayers to eval mode
        self.self_attention.set_eval();
        self.feed_forward.set_eval();
        self.layer_norm.set_eval();
        self.feed_forward_layer_norm.set_eval();
    }

    fn get_params(&mut self) -> Vec<crate::layers::ParamHandle> {
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
