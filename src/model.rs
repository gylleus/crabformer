use std::{sync::Arc, time::Instant};

use clap::builder::StringValueParser;
use ndarray::{Array2, Array3, s};
use parking_lot::Mutex;
use rand::distr::{Distribution, weighted::WeightedIndex};
use serde::{Deserialize, Serialize};

use crate::{
    adamw::{self, AdamWOptimizer, ParamHandle},
    data::{Batch, DataLoader},
    errors::ModelError,
    layers::{
        Layer, Layer3Df32, LayerCacheParam, ZeroGrad,
        dropout::Dropout,
        embedding::EmbeddingLayer,
        linear::LinearLayer,
        normalization::{LayerNormLayer, Softmax},
        transformer_block::TransformerBlock,
    },
    loss::{cross_entropy_loss, cross_entropy_loss_backward},
    metrics::{TrainingMetrics, TrainingMetricsHandle},
    params::{
        ADAMW_BETA1, ADAMW_BETA2, ADAMW_EPSILON, ATTENTION_HEADS, BATCH_SIZE, CHECKPOINTS_DIR,
        DROPOUT_RATE, EMBED_DIMENSION, FF_HIDDEN_DIMENSION, GLOBAL_RNG, LEARNING_RATE,
        SAVE_EVERY_N_STEPS, SEQUENCE_LENGTH, TEMPERATURE, TRANSFORMER_BLOCKS, WEIGHT_DECAY,
    },
};

#[derive(Serialize, Deserialize)]
pub struct CrabformerModel {
    token_embedding_layer: EmbeddingLayer,
    position_embedding_layer: EmbeddingLayer,
    transformer_layers: Vec<TransformerBlock>,
    final_layer_norm: LayerNormLayer,
    // Final layer to project to vocabulary size (no weight tying to reuse input embeddings layer)
    output_layer: LinearLayer,

    // Training mode flag
    training: bool,
    // Cache for backward pass
    #[serde(skip)]
    last_embeddings: LayerCacheParam<Array3<f32>>,
    #[serde(skip)]
    last_positions: LayerCacheParam<Array2<u32>>,
    #[serde(skip)]
    adamw_optimizer: Option<AdamWOptimizer>,
}

impl CrabformerModel {
    pub fn new(vocab_size: usize, seed: Option<u64>) -> Result<Self, ModelError> {
        let transformer_layers: Vec<_> = (0..TRANSFORMER_BLOCKS)
            .map(|i| {
                let name = format!("TransformerBlock_{}", i);
                TransformerBlock::new(
                    EMBED_DIMENSION,
                    ATTENTION_HEADS,
                    FF_HIDDEN_DIMENSION,
                    DROPOUT_RATE,
                    Some(name),
                )
            })
            .collect::<Result<Vec<TransformerBlock>, ModelError>>()?;

        Ok(Self {
            token_embedding_layer: EmbeddingLayer::new(
                vocab_size,
                EMBED_DIMENSION,
                Some("TokenEmbeddingLayer".into()),
            ),
            position_embedding_layer: EmbeddingLayer::new(
                SEQUENCE_LENGTH,
                EMBED_DIMENSION,
                Some("PositionEmbeddingLayer".into()),
            ),
            transformer_layers,
            final_layer_norm: crate::layers::normalization::LayerNormLayer::new(
                EMBED_DIMENSION,
                Some("FinalLayerNorm".into()),
            ),
            output_layer: LinearLayer::new(EMBED_DIMENSION, vocab_size, Some("OutputLayer".into())),

            training: false,
            adamw_optimizer: Some(AdamWOptimizer::new(
                LEARNING_RATE,
                ADAMW_BETA1,
                ADAMW_BETA2,
                ADAMW_EPSILON,
                WEIGHT_DECAY,
            )),
            last_embeddings: LayerCacheParam::new("CrabformerModel::last_embeddings".to_string()),
            last_positions: LayerCacheParam::new("CrabformerModel::last_positions".to_string()),
        })
    }

    pub fn next_token_batch(&self, input: &Batch) -> Vec<u32> {
        let model_output = self.forward_batch(input);
        let (batch_size, seq_length, _vocab_size) = model_output.dim();
        assert_eq!(seq_length, SEQUENCE_LENGTH);

        let mut next_tokens = Vec::with_capacity(batch_size);

        let mut rng = GLOBAL_RNG.lock();
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

        // Apply dropout to embeddings (only during training)
        if self.training {
            token_embedding_output.apply_dropout(DROPOUT_RATE);
        }

        let mut output = token_embedding_output;
        for layer in &self.transformer_layers {
            output = layer.forward(&output);
        }

        output = self.final_layer_norm.forward(&output);

        // Final output layer to get logits for each token in the vocabulary
        output = self.output_layer.forward(&output);

        output
    }

    /// Forward pass with caching for training
    pub fn forward_train(
        &mut self,
        input: &Batch,
        metrics_handle: TrainingMetricsHandle,
    ) -> Array3<f32> {
        let token_embedding_start = Instant::now();
        let mut token_embedding_output = self.token_embedding_layer.forward(&input.x);
        let token_embedding_duration = token_embedding_start.elapsed();

        let (batch_size, sequence_length) = input.x.dim();

        let positions = Array2::from_shape_fn((batch_size, sequence_length), |(_, j)| {
            j as u32 // Each position in the sequence gets its index
        });

        let position_embedding_start = Instant::now();
        let position_embedding_output = self.position_embedding_layer.forward(&positions);
        let position_embedding_duration = position_embedding_start.elapsed();

        token_embedding_output += &position_embedding_output;

        // Apply dropout to embeddings
        token_embedding_output.apply_dropout(DROPOUT_RATE);

        // Cache for backward pass
        *self.last_embeddings.mut_ref() = Some(token_embedding_output.clone());
        *self.last_positions.mut_ref() = Some(positions);

        let mut output = token_embedding_output;
        for layer in &mut self.transformer_layers {
            output = layer.forward(&output);
        }

        output = self.final_layer_norm.forward(&output);

        // Final output layer to get logits for each token in the vocabulary
        let output_start = Instant::now();
        output = self.output_layer.forward(&output);
        let output_duration = output_start.elapsed();

        {
            let mut metrics = metrics_handle.lock();
            metrics.token_embedding_duration.forward_duration += token_embedding_duration;
            metrics.positional_embedding_duration.forward_duration += position_embedding_duration;
            metrics.output_layer_duration.forward_duration += output_duration;
        }

        output
    }

    /// Backward pass: propagates gradients through the model
    ///
    /// # Arguments
    /// * `grad_output` - Gradient of loss w.r.t. model output [batch_size, seq_len, vocab_size]
    ///
    /// # Returns
    /// Result indicating success or error
    pub fn backward(
        &mut self,
        grad_output: &Array3<f32>,
        metrics_handle: TrainingMetricsHandle,
    ) -> Result<(), ModelError> {
        // Backprop through output layer
        let output_start = Instant::now();
        let mut grad = self.output_layer.backward(grad_output)?;
        let output_duration = output_start.elapsed();

        // Backprop through final layer norm
        grad = self.final_layer_norm.backward(&grad)?;

        // Backprop through transformer layers in reverse
        for layer in self.transformer_layers.iter_mut().rev() {
            grad = layer.backward(&grad)?;
        }

        // Backprop through dropout (approximation: pass through)

        // Backprop through position embeddings
        let position_embedding_start = Instant::now();
        self.position_embedding_layer.backward(&grad)?;
        let position_embedding_duration = position_embedding_start.elapsed();

        // Backprop through token embeddings
        // The gradient is the same for both embedding layers since they're added together
        let token_embedding_start = Instant::now();
        self.token_embedding_layer.backward(&grad)?;
        let token_embedding_duration = token_embedding_start.elapsed();

        {
            let mut metrics = metrics_handle.lock();

            metrics.token_embedding_duration.backward_duration += token_embedding_duration;
            metrics.positional_embedding_duration.backward_duration += position_embedding_duration;
            metrics.output_layer_duration.backward_duration += output_duration;
        }

        Ok(())
    }

    /// Set model to training mode
    fn set_training(&mut self, training: bool, metrics_handle: TrainingMetricsHandle) {
        self.training = training;
        for layer in &mut self.transformer_layers {
            if training {
                layer.set_train(metrics_handle.clone());
            } else {
                layer.set_eval();
            }
        }

        // Set training mode for embedding and output layers
        if training {
            self.token_embedding_layer.set_train(metrics_handle.clone());
            self.position_embedding_layer
                .set_train(metrics_handle.clone());
            self.final_layer_norm.set_train(metrics_handle.clone());
            self.output_layer.set_train(metrics_handle.clone());
        } else {
            self.token_embedding_layer.set_eval();
            self.position_embedding_layer.set_eval();
            self.final_layer_norm.set_eval();
            self.output_layer.set_eval();
        }
    }

    /// Zero out all gradients
    pub fn zero_grad(&mut self) {
        self.token_embedding_layer.zero_grad();
        self.position_embedding_layer.zero_grad();
        self.final_layer_norm.zero_grad();
        self.output_layer.zero_grad();

        // Zero grad for transformer blocks
        for layer in &mut self.transformer_layers {
            layer.zero_grad();
        }
    }

    /// Training loop
    pub fn train(
        &mut self,
        data_loader: &mut DataLoader,
        num_epochs: usize,
    ) -> Result<(), ModelError> {
        let metrics = TrainingMetrics::new();
        let metrics_handle = Arc::new(Mutex::new(metrics));

        self.set_training(true, metrics_handle.clone());

        for epoch in 0..num_epochs {
            let mut total_loss = 0.0;
            let mut num_batches = 0;

            // Reset data loader for new epoch
            // TODO
            // data_loader.reset();

            while let Some(batch) = data_loader
                .next_batch()
                .map_err(|e| ModelError::DataError(e))?
            {
                // Set all gradients to zero
                self.zero_grad();

                // Forward pass
                let logits = self.forward_train(&batch, metrics_handle.clone());

                // Compute loss
                let loss = cross_entropy_loss(&logits, &batch.y);
                total_loss += loss;
                num_batches += 1;

                // Compute gradients
                let grad_logits = cross_entropy_loss_backward(&logits, &batch.y);

                // Backward pass
                self.backward(&grad_logits, metrics_handle.clone())?;

                // Update parameters using AdamW optimizer
                let Some(optimizer) = &mut self.adamw_optimizer else {
                    return Err(ModelError::OptimizerError(
                        "AdamW optimizer not initialized".to_string(),
                    ));
                };

                let mut params: Vec<ParamHandle> = self
                    .token_embedding_layer
                    .get_params()
                    .into_iter()
                    .chain(self.position_embedding_layer.get_params())
                    .chain(self.final_layer_norm.get_params())
                    .chain(self.output_layer.get_params())
                    .chain(
                        self.transformer_layers
                            .iter_mut()
                            .flat_map(|layer| layer.get_params()),
                    )
                    .collect();

                optimizer.step(&mut params);

                // Update training metrics

                if num_batches % 10 == 0 {
                    println!("Epoch {}, Batch {}, Loss: {:.4}", epoch, num_batches, loss);
                }

                if num_batches % SAVE_EVERY_N_STEPS == 0 {
                    self.save_weights(epoch, num_batches)?;
                }

                {
                    let mut metrics = metrics_handle.lock();
                    metrics.processed_batches += 1;
                    metrics.current_loss = loss;

                    // Display metrics
                    println!("Metrics: {:?}", metrics);

                    // Reset durations for next batch
                    metrics.reset_durations();
                }
            }

            let avg_loss = total_loss / num_batches as f32;
            println!("Epoch {} completed. Average loss: {:.4}", epoch, avg_loss);
        }

        Ok(())
    }

    fn save_weights(&self, epoch: usize, step: usize) -> Result<(), ModelError> {
        let serialized = ron::to_string(self)
            .map_err(|e| ModelError::SerializationError(format!("Serialization failed: {}", e)))?;

        let file_name = format!(
            "{}/checkpoint_epoch_{}_{}.ron",
            CHECKPOINTS_DIR, epoch, step
        );

        // Ensure checkpoints directory exists
        std::fs::create_dir_all(CHECKPOINTS_DIR).map_err(|e| {
            ModelError::IOError(format!("Failed to create checkpoints directory: {}", e))
        })?;

        std::fs::write(file_name, serialized)
            .map_err(|e| ModelError::IOError(format!("Failed to write model to file: {}", e)))?;

        println!(
            "Model weights saved to {}/model_epoch_{}_{}.ron",
            CHECKPOINTS_DIR, epoch, step
        );

        Ok(())
    }
}
