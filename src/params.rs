use std::sync::atomic::{AtomicU64, Ordering};

use rand::{SeedableRng, rngs::StdRng};

pub const BATCH_SIZE: usize = 16;
pub const TEMPERATURE: f32 = 0.9;

/// Dimension of the hidden layer in the feed-forward layers for the transformer blocks.
pub const FF_HIDDEN_DIMENSION: usize = 16;
pub const ATTENTION_HEADS: usize = 4;
pub const TRANSFORMER_BLOCKS: usize = 4;
/// The length of each training sequence (T tokens).
pub const SEQUENCE_LENGTH: usize = 64;
pub const EMBED_DIMENSION: usize = 32;

/// The size of the data ring buffer used for training. A larger size improves batch shuffling.
pub const RING_BUFFER_SIZE: usize = 16 * SEQUENCE_LENGTH;
/// The number of new tokens to add to the ring buffer after reading each batch.
pub const RING_BUFFER_ADVANCE: usize = SEQUENCE_LENGTH;

pub const QKV_BIAS: bool = false;
pub const DROPOUT_RATE: f32 = 0.1;

pub const LEARNING_RATE: f32 = 0.001;
pub const WEIGHT_DECAY: f32 = 0.01;
pub const ADAMW_BETA1: f32 = 0.9;
pub const ADAMW_BETA2: f32 = 0.999;
pub const ADAMW_EPSILON: f32 = 1e-8;

pub fn get_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_os_rng(),
    }
}

pub struct MutableRng {
    seed_counter: Option<AtomicU64>,
}

impl MutableRng {
    pub fn new(seed: Option<u64>) -> Self {
        Self {
            seed_counter: seed.map(AtomicU64::new),
        }
    }

    pub fn get_rng(&self) -> StdRng {
        if let Some(counter) = &self.seed_counter {
            // Get and increment the internal RNG seed
            let current_count = counter.fetch_add(1, Ordering::Relaxed);
            get_rng(Some(current_count))
        } else {
            get_rng(None)
        }
    }
}
