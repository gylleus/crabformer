use ndarray::{Array2, Array3, s};
use rand::distr::{Distribution, weighted::WeightedIndex};

use crate::{
    data::Batch,
    errors::ModelError,
    layers::{
        Layer, dropout::Dropout, embedding::EmbeddingLayer, linear::LinearLayer,
        normalization::Softmax, transformer_block::TransformerBlock,
    },
    params::{
        ATTENTION_HEADS, BATCH_SIZE, DROPOUT_RATE, EMBED_DIMENSION, FF_HIDDEN_DIMENSION,
        MutableRng, SEQUENCE_LENGTH, TEMPERATURE, TRANSFORMER_BLOCKS, get_rng,
    },
};

pub struct CrabformerModel {
    token_embedding_layer: EmbeddingLayer,
    position_embedding_layer: EmbeddingLayer,
    layers: Vec<Box<dyn Layer>>,
    // Final layer to project to vocabulary size (no weight tying to reuse input embeddings layer)
    output_layer: LinearLayer,
    seed: Option<u64>,
    rng: MutableRng,
}

impl CrabformerModel {
    pub fn new(vocab_size: usize, seed: Option<u64>) -> Result<Self, ModelError> {
        let transformer_block = || -> Result<Box<TransformerBlock>, ModelError> {
            let block = TransformerBlock::new(
                EMBED_DIMENSION,
                ATTENTION_HEADS,     // num_heads
                FF_HIDDEN_DIMENSION, // dim_ff
                DROPOUT_RATE,
                seed,
            )?;
            Ok(Box::new(block))
        };

        let mut layers: Vec<Box<dyn Layer>> = (0..TRANSFORMER_BLOCKS)
            .map(|_| transformer_block().map(|b| b as Box<dyn Layer>))
            .collect::<Result<Vec<Box<dyn Layer>>, ModelError>>()?;

        // Add final layer norm
        layers.push(Box::new(crate::layers::normalization::LayerNormLayer::new(
            EMBED_DIMENSION,
        )));

        Ok(Self {
            token_embedding_layer: EmbeddingLayer::new(vocab_size, EMBED_DIMENSION, seed),
            position_embedding_layer: EmbeddingLayer::new(SEQUENCE_LENGTH, EMBED_DIMENSION, seed),
            layers,
            output_layer: LinearLayer::new(EMBED_DIMENSION, vocab_size, seed),
            seed,
            rng: MutableRng::new(seed),
        })
    }

    pub fn next_token_batch(&self, input: &Batch) -> Vec<u32> {
        let model_output = self.forward_batch(input);
        let (batch_size, seq_length, _vocab_size) = model_output.dim();
        assert_eq!(seq_length, SEQUENCE_LENGTH);

        let mut next_tokens = Vec::with_capacity(batch_size);
        let mut rng = get_rng(self.seed);
        for batch_idx in 0..batch_size {
            // Get the logits for the last token in the sequence
            let mut last_token_logits = model_output
                .slice(s![batch_idx, seq_length - 1, ..])
                .to_owned();

            // Apply softmax to convert logits to probabilities
            last_token_logits.softmax(0, Some(TEMPERATURE));

            let dist = WeightedIndex::new(&last_token_logits).unwrap();
            let predicted_token = dist.sample(&mut rng);

            next_tokens.push(predicted_token as u32);
        }

        next_tokens
    }

    pub fn forward_batch(&self, input: &Batch) -> Array3<f32> {
        let mut token_embedding_output = self.token_embedding_layer.forward(&input.x);

        let (batch_size, sequence_length) = input.x.dim();

        let positions = Array2::from_shape_fn((batch_size, sequence_length), |(_, j)| {
            j as u32 // Each position in the sequence gets its index
        });

        let position_embedding_output = self.position_embedding_layer.forward(&positions);

        token_embedding_output += &position_embedding_output;

        // Apply dropout to embeddings
        token_embedding_output.apply_dropout(DROPOUT_RATE, &mut get_rng(self.seed));

        let mut output = token_embedding_output;
        for layer in &self.layers {
            output = layer.forward(&output);
        }

        // Final output layer to get logits for each token in the vocabulary
        output = self.output_layer.forward_3d(&output);

        output
    }
}
