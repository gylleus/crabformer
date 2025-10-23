use ndarray::Array3;

use crate::{
    errors::ModelError,
    layers::{
        Layer, LayerCacheParam, ZeroGrad, dropout::Dropout, linear::FeedForwardLayer,
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

    // Training mode flag
    training: bool,

    // Cache for backward pass
    last_input: LayerCacheParam<Array3<f32>>,
    last_attention_output: LayerCacheParam<Array3<f32>>,
    last_layer_norm_input: LayerCacheParam<Array3<f32>>,
    last_layer_norm_attention: LayerCacheParam<Array3<f32>>,
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
            training: false,
            last_input: LayerCacheParam::new("TransformerBlock::last_input"),
            last_attention_output: LayerCacheParam::new("TransformerBlock::last_attention_output"),
            last_layer_norm_input: LayerCacheParam::new("TransformerBlock::last_layer_norm_input"),
            last_layer_norm_attention: LayerCacheParam::new(
                "TransformerBlock::last_layer_norm_attention",
            ),
        })
    }
}

impl Layer for TransformerBlock {
    type Input = Array3<f32>;
    type Output = Array3<f32>;

    fn forward(&self, input: &Self::Input) -> Self::Output {
        let layer_normalized_input = self.layer_norm.forward(input);

        // Cache input if in training mode
        if self.training {
            *self.last_input.mut_ref() = Some(input.clone());
            *self.last_layer_norm_input.mut_ref() = Some(layer_normalized_input.clone());
        }

        // Self-attention sub-layer
        let mut attention_output = self.self_attention.forward(&layer_normalized_input);
        // .to_array3()?;

        // Apply dropout after attention
        attention_output.apply_dropout(self.dropout_rate, &mut self.mutable_rng.get_rng());

        // Residual connection (skip connection)
        attention_output += input;

        // Normalize again before feed-forward
        let layer_normalized_attention = self.layer_norm.forward(&attention_output);

        // Cache attention output if in training mode
        if self.training {
            *self.last_layer_norm_attention.mut_ref() = Some(layer_normalized_attention.clone());
            *self.last_attention_output.mut_ref() = Some(attention_output.clone());
        }

        // Feed-forward sub-layer
        let mut feed_forward_output = self.feed_forward.forward(&layer_normalized_attention);

        // Apply dropout after feed-forward
        feed_forward_output.apply_dropout(self.dropout_rate, &mut self.mutable_rng.get_rng());

        // Second residual connection
        attention_output += &feed_forward_output;

        attention_output
    }

    fn backward(&mut self, grad_output: &Self::Output) -> Result<Self::Input, ModelError> {
        // Get cached values
        let input = self.last_input.read_ref()?;
        let layer_norm_input = self.last_layer_norm_input.read_ref()?;
        let attention_output = self.last_attention_output.read_ref()?;
        let layer_norm_attention = self.last_layer_norm_attention.read_ref()?;

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
        let mut grad_layer_norm_attention = self.feed_forward.backward(&grad_ff_output)?;

        // Backprop through second layer norm
        grad_layer_norm_attention += &self.layer_norm.backward(&grad_attention_after_residual)?;

        // Backprop through first residual connection
        // attention_output = attention + input
        // Gradients flow to both paths
        // let mut grad_attention = grad_layer_norm_attention.clone();
        // let mut grad_input_from_residual = grad_layer_norm_attention.clone();

        // Backprop through dropout after attention (approximation: pass through)

        // Backprop through self-attention
        let grad_layer_norm_input = self.self_attention.backward(&grad_layer_norm_attention)?;

        // Backprop through first layer norm
        let grad_input_from_norm = self.layer_norm.backward(&grad_layer_norm_input)?;

        // Combine gradients from both residual paths
        let grad_input = grad_input_from_norm + &grad_layer_norm_attention;

        Ok(grad_input)
    }

    fn set_train(&mut self) {
        self.training = true;

        // Set sublayers to training mode
        self.self_attention.set_train();
        self.feed_forward.set_train();
        self.layer_norm.set_train();
    }

    fn set_eval(&mut self) {
        self.training = false;

        // Free cache memory
        self.last_input.clear();
        self.last_attention_output.clear();
        self.last_layer_norm_input.clear();
        self.last_layer_norm_attention.clear();

        // Set sublayers to eval mode
        self.self_attention.set_eval();
    }
}

impl ZeroGrad for TransformerBlock {
    fn zero_grad(&mut self) {
        self.self_attention.zero_grad();
        self.feed_forward.zero_grad();
        self.layer_norm.zero_grad();
    }
}
