use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::Mutex;

pub type TrainingMetricsHandle = Arc<Mutex<TrainingMetrics>>;

#[derive(Debug, Default)]
pub struct TrainingDuration {
    pub forward_duration_total: Duration,
    pub backward_duration_total: Duration,
    pub forward_count: usize,
    pub backward_count: usize,
}

impl TrainingDuration {
    pub fn add_forward(&mut self, duration: Duration) {
        self.forward_duration_total += duration;
        self.forward_count += 1;
    }

    pub fn add_backward(&mut self, duration: Duration) {
        self.backward_duration_total += duration;
        self.backward_count += 1;
    }

    pub fn avg_forward_duration(&self) -> Duration {
        if self.forward_count > 0 {
            self.forward_duration_total / self.forward_count as u32
        } else {
            Duration::ZERO
        }
    }

    pub fn avg_backward_duration(&self) -> Duration {
        if self.backward_count > 0 {
            self.backward_duration_total / self.backward_count as u32
        } else {
            Duration::ZERO
        }
    }
}

#[derive(Debug)]
pub struct TrainingMetrics {
    pub epochs: usize,
    pub batches_per_epoch: usize,

    pub current_epoch: usize,
    pub processed_batches: usize,

    pub start_time: Instant,

    /// History of (batch_number, loss) for plotting
    pub loss_history: Vec<(usize, f32)>,

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
    pub fn new(epochs: usize, batches_per_epoch: usize) -> Self {
        Self {
            epochs,
            batches_per_epoch,
            current_epoch: 0,
            processed_batches: 0,
            start_time: Instant::now(),
            loss_history: Vec::new(),
            transformer_block_duration: TrainingDuration::default(),
            token_embedding_duration: TrainingDuration::default(),
            positional_embedding_duration: TrainingDuration::default(),
            attention_duration: TrainingDuration::default(),
            feed_forward_duration: TrainingDuration::default(),
            layer_norm_duration: TrainingDuration::default(),
            output_layer_duration: TrainingDuration::default(),
        }
    }

    pub fn current_loss(&self) -> f32 {
        self.loss_history
            .last()
            .map(|(_, loss)| *loss)
            .unwrap_or(0.0)
    }
}
