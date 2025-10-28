use std::{sync::Arc, time::Instant};

use ndarray::{Array2, Array3};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::{
    adamw::{AdamWOptimizer, ParamHandle},
    data::{Batch, DataLoader},
    errors::ModelError,
    layers::{
        Layer, LayerCacheParam, ZeroGrad, dropout::Dropout, embedding::EmbeddingLayer,
        linear::LinearLayer, normalization::LayerNormLayer, transformer_block::TransformerBlock,
    },
    loss::{cross_entropy_loss, cross_entropy_loss_backward},
    metrics::{TrainingMetrics, TrainingMetricsHandle},
};

#[derive(Serialize, Deserialize)]
pub struct CrabformerModel {
    pub config: ModelConfig,

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
    embedding_dropout_mask: LayerCacheParam<Vec<bool>>,
    #[serde(skip)]
    adamw_optimizer: Option<AdamWOptimizer>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub transformer_blocks: usize,
    pub embed_dim: usize,
    pub attention_heads: usize,
    pub ff_hidden_dim: usize,
    pub dropout: f32,
    pub seq_length: usize,

    pub qkv_bias: bool,
    pub learning_rate: f32,
    pub weight_decay: f32,
}

impl CrabformerModel {
    pub fn new(config: ModelConfig) -> Result<Self, ModelError> {
        let transformer_layers: Vec<_> = (0..config.transformer_blocks)
            .map(|i| {
                let name = format!("TransformerBlock_{}", i);
                TransformerBlock::new(
                    config.seq_length,
                    config.embed_dim,
                    config.attention_heads,
                    config.ff_hidden_dim,
                    config.dropout,
                    config.qkv_bias,
                    Some(name),
                )
            })
            .collect::<Result<Vec<TransformerBlock>, ModelError>>()?;

        Ok(Self {
            token_embedding_layer: EmbeddingLayer::new(
                config.vocab_size,
                config.embed_dim,
                Some("TokenEmbeddingLayer".into()),
            ),
            position_embedding_layer: EmbeddingLayer::new(
                config.seq_length,
                config.embed_dim,
                Some("PositionEmbeddingLayer".into()),
            ),
            transformer_layers,
            final_layer_norm: crate::layers::normalization::LayerNormLayer::new(
                config.embed_dim,
                Some("FinalLayerNorm".into()),
            ),
            output_layer: LinearLayer::new(
                config.embed_dim,
                config.vocab_size,
                Some("OutputLayer".into()),
            )
            .with_bias(),

            training: false,
            adamw_optimizer: Some(AdamWOptimizer::new(
                config.learning_rate,
                0.9,
                0.999,
                1e-8,
                config.weight_decay,
            )),
            last_embeddings: LayerCacheParam::new("CrabformerModel::last_embeddings".to_string()),
            last_positions: LayerCacheParam::new("CrabformerModel::last_positions".to_string()),
            embedding_dropout_mask: LayerCacheParam::new(
                "CrabformerModel::embedding_dropout_mask".to_string(),
            ),
            config,
        })
    }

    /// Forward pass without caching (for inference)
    pub fn forward(&self, input: &Array2<u32>) -> Array3<f32> {
        let mut token_embedding_output = self.token_embedding_layer.forward(input);

        let (batch_size, sequence_length) = input.dim();

        let positions = Array2::from_shape_fn((batch_size, sequence_length), |(_, j)| {
            j as u32 // Each position in the sequence gets its index
        });

        let position_embedding_output = self.position_embedding_layer.forward(&positions);

        token_embedding_output += &position_embedding_output;

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
    fn forward_train(
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

        // Apply dropout to embeddings and save mask
        let dropout_mask = token_embedding_output.apply_dropout(self.config.dropout);
        *self.embedding_dropout_mask.mut_ref() = dropout_mask;

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
            metrics
                .token_embedding_duration
                .add_forward(token_embedding_duration);
            metrics
                .positional_embedding_duration
                .add_forward(position_embedding_duration);
            metrics.output_layer_duration.add_forward(output_duration);
        }

        output
    }

    /// Backward pass: propagates gradients through the model
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

        // Backprop through dropout - apply the same mask from forward pass
        if let Some(ref mask) = *self.embedding_dropout_mask.mut_ref() {
            grad.apply_dropout_backward(mask, self.config.dropout);
        }

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

            metrics
                .token_embedding_duration
                .add_backward(token_embedding_duration);
            metrics
                .positional_embedding_duration
                .add_backward(position_embedding_duration);
            metrics.output_layer_duration.add_backward(output_duration);
        }

        Ok(())
    }

    /// Set model to training mode
    pub fn set_training(&mut self, metrics_handle: TrainingMetricsHandle) {
        for layer in &mut self.transformer_layers {
            layer.set_train(metrics_handle.clone());
        }
        self.token_embedding_layer.set_train(metrics_handle.clone());
        self.position_embedding_layer
            .set_train(metrics_handle.clone());
        self.final_layer_norm.set_train(metrics_handle.clone());
        self.output_layer.set_train(metrics_handle.clone());
    }

    pub fn set_eval(&mut self) {
        for layer in &mut self.transformer_layers {
            layer.set_eval();
        }
        self.token_embedding_layer.set_eval();
        self.position_embedding_layer.set_eval();
        self.final_layer_norm.set_eval();
        self.output_layer.set_eval();
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
        save_every_n_steps: usize,
        out_dir: &str,
    ) -> Result<(), ModelError> {
        let batches_per_epoch = data_loader.total_batches();

        let metrics =
            TrainingMetrics::new(num_epochs, batches_per_epoch, self.transformer_layers.len());
        let metrics_handle = Arc::new(Mutex::new(metrics));

        self.set_training(metrics_handle.clone());

        // Start dashboard in a separate thread
        let (quit_dashboard, dashboard_handle) =
            crate::dashboard::start_dashboard(metrics_handle.clone());

        let mut total_steps = 0;
        for epoch in 0..num_epochs {
            let mut step = 0;

            data_loader.reset()?;

            while let Some(batch) = data_loader.next_batch().map_err(ModelError::DataError)? {
                // Set all gradients to zero
                self.zero_grad();

                // Forward pass
                let logits = self.forward_train(&batch, metrics_handle.clone());

                // Compute loss
                let loss = cross_entropy_loss(&logits, &batch.y);
                step += 1;
                total_steps += 1;

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

                // Get all model parameters to optimize from layers
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

                // Perform optimization step on model parameters
                optimizer.step(&mut params);

                // Update training metrics
                {
                    let mut metrics = metrics_handle.lock();
                    metrics.processed_batches += 1;
                    let batch_num = metrics.processed_batches;
                    metrics.loss_history.push((batch_num, loss));
                }

                if total_steps % save_every_n_steps == 0 {
                    let file_name = format!("checkpoint_epoch_{}_{}.ron", epoch, step);

                    self.save_weights(out_dir, &file_name)?;
                }

                // Check for dashboard quit signal
                if *quit_dashboard.lock() {
                    println!("Training interrupted by user. Exiting...");
                    return Ok(());
                }
            }

            // Update current epoch in metrics
            {
                let mut metrics = metrics_handle.lock();
                metrics.current_epoch = epoch + 1;
            }
        }

        // Stop dashboard thread
        *quit_dashboard.lock() = true;
        // Wait for dashboard thread to properly exit
        dashboard_handle.join().unwrap();

        Ok(())
    }

    pub fn save_weights(&self, dir: &str, file_name: &str) -> Result<(), ModelError> {
        let serialized = ron::to_string(self)
            .map_err(|e| ModelError::SerializationError(format!("Serialization failed: {}", e)))?;

        // Ensure checkpoints directory exists
        std::fs::create_dir_all(dir).map_err(|e| {
            ModelError::IOError(format!("Failed to create checkpoints directory: {}", e))
        })?;

        std::fs::write(format!("{}/{}", dir, file_name), serialized)
            .map_err(|e| ModelError::IOError(format!("Failed to write model to file: {}", e)))?;

        Ok(())
    }

    pub fn load(checkpoint_path: String) -> Result<Self, ModelError> {
        let data = std::fs::read_to_string(&checkpoint_path).map_err(|e| {
            ModelError::IOError(format!(
                "Failed to read checkpoint file '{}': {}",
                checkpoint_path, e
            ))
        })?;

        let model: CrabformerModel = ron::from_str(&data).map_err(|e| {
            ModelError::SerializationError(format!(
                "Failed to deserialize model from checkpoint: {}",
                e
            ))
        })?;

        Ok(model)
    }
}
