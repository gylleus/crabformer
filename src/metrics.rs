use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::Mutex;

pub type TrainingMetricsHandle = Arc<Mutex<TrainingMetrics>>;

#[derive(Debug, Default)]
pub struct TrainingDuration {
    pub forward_duration: Duration,
    pub backward_duration: Duration,
}

impl TrainingDuration {
    pub fn reset(&mut self) {
        self.forward_duration = Duration::ZERO;
        self.backward_duration = Duration::ZERO;
    }

    pub fn total_duration(&self) -> Duration {
        self.forward_duration + self.backward_duration
    }
}

#[derive(Debug)]
pub struct TrainingMetrics {
    pub current_loss: f32,
    pub processed_batches: usize,

    pub start_time: Instant,

    /// Total time spent in transformer blocks
    pub transformer_block_duration: TrainingDuration,

    pub token_embedding_duration: TrainingDuration,
    pub positional_embedding_duration: TrainingDuration,
    pub attention_duration: TrainingDuration,
    pub feed_forward_duration: TrainingDuration,

    pub layer_norm_duration: TrainingDuration,

    pub output_layer_duration: TrainingDuration,
}

impl TrainingMetrics {
    pub fn new() -> Self {
        Self {
            current_loss: 0.0,
            processed_batches: 0,
            start_time: Instant::now(),
            transformer_block_duration: TrainingDuration::default(),
            token_embedding_duration: TrainingDuration::default(),
            positional_embedding_duration: TrainingDuration::default(),
            attention_duration: TrainingDuration::default(),
            feed_forward_duration: TrainingDuration::default(),
            layer_norm_duration: TrainingDuration::default(),
            output_layer_duration: TrainingDuration::default(),
        }
    }

    pub fn reset_durations(&mut self) {
        self.transformer_block_duration.reset();
        self.token_embedding_duration.reset();
        self.positional_embedding_duration.reset();
        self.attention_duration.reset();
        self.feed_forward_duration.reset();
        self.layer_norm_duration.reset();
        self.output_layer_duration.reset();
    }
}
