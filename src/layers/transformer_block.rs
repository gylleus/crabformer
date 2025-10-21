use ndarray::Array3;

use crate::{
    errors::ModelError,
    layers::{
        Layer, dropout::Dropout, linear::FeedForwardLayer,
        multi_head_attention::MultiHeadAttentionLayer, normalization::LayerNormLayer,
    },
    params::MutableRng,
};

pub struct TransformerBlock {
    pub self_attention: MultiHeadAttentionLayer,
    pub feed_forward: FeedForwardLayer,
    pub layer_norm: LayerNormLayer,
    pub dropout_rate: f32,
    pub mutable_rng: MutableRng,
}

impl TransformerBlock {
    pub fn new(
        dim_model: usize,
        num_heads: usize,
        dim_ff: usize,
        dropout_rate: f32,
        seed: Option<u64>,
    ) -> Result<Self, ModelError> {
        let attention_layer =
            MultiHeadAttentionLayer::new(dim_model, dim_model, num_heads, dropout_rate, seed)?;

        Ok(Self {
            self_attention: attention_layer,
            feed_forward: FeedForwardLayer::new(dim_model, dim_ff, seed),
            layer_norm: LayerNormLayer::new(dim_model),
            dropout_rate,
            mutable_rng: MutableRng::new(seed),
        })
    }
}

impl Layer for TransformerBlock {
    fn forward(&self, input: &Array3<f32>) -> Array3<f32> {
        let layer_normalized_input = self.layer_norm.forward(input);
        // Self-attention sub-layer
        let mut attention_output = self.self_attention.forward(&layer_normalized_input);

        // Apply dropout after attention
        attention_output.apply_dropout(self.dropout_rate, &mut self.mutable_rng.get_rng());

        // Residual connection (skip connection)
        attention_output += input;

        // Normalize again before feed-forward
        let layer_normalized_attention = self.layer_norm.forward(&attention_output);

        // Feed-forward sub-layer
        let mut feed_forward_output = self.feed_forward.forward_3d(&layer_normalized_attention);

        // Apply dropout after feed-forward
        feed_forward_output.apply_dropout(self.dropout_rate, &mut self.mutable_rng.get_rng());

        // Second residual connection
        attention_output += &feed_forward_output;

        attention_output
    }
}
