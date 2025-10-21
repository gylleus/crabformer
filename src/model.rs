use ndarray::{Array2, Array3};

use crate::{
    data::Batch,
    errors::ModelError,
    layers::{
        Layer, dropout::Dropout, embedding::EmbeddingLayer, transformer_block::TransformerBlock,
    },
    params::{BATCH_SIZE, DROPOUT_RATE, EMBED_DIMENSION, get_rng},
};

pub struct CrabformerModel {
    token_embedding_layer: EmbeddingLayer,
    position_embedding_layer: EmbeddingLayer,
    layers: Vec<Box<dyn Layer>>,
    seed: Option<u64>,
}

impl CrabformerModel {
    pub fn new(vocab_size: usize, seed: Option<u64>) -> Result<Self, ModelError> {
        let transformer_block = || -> Result<Box<TransformerBlock>, ModelError> {
            let block = TransformerBlock::new(
                EMBED_DIMENSION,
                2,  // num_heads
                32, // dim_ff
                DROPOUT_RATE,
                seed,
            )?;
            Ok(Box::new(block))
        };

        let layers: Vec<Box<dyn Layer>> = vec![
            transformer_block()?,
            transformer_block()?,
            transformer_block()?,
        ];

        Ok(Self {
            token_embedding_layer: EmbeddingLayer::new(vocab_size, EMBED_DIMENSION, seed),
            position_embedding_layer: EmbeddingLayer::new(512, EMBED_DIMENSION, seed),
            layers,
            seed,
        })
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
        output
    }
}
