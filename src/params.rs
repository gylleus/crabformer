use std::sync::atomic::{AtomicU64, Ordering};

use rand::{SeedableRng, rngs::StdRng};

pub const BATCH_SIZE: usize = 16;

/// The length of each training sequence (T tokens).
pub const SEQUENCE_LENGTH: usize = 4;
/// The size of the data ring buffer used for training. A larger size improves batch shuffling.
pub const RING_BUFFER_SIZE: usize = 16 * SEQUENCE_LENGTH;
/// The number of new tokens to add to the ring buffer after reading each batch.
pub const RING_BUFFER_ADVANCE: usize = SEQUENCE_LENGTH;

pub const EMBED_DIMENSION: usize = 8;

pub const QKV_BIAS: bool = false;
pub const DROPOUT_RATE: f32 = 0.1;

/// Dimension of the hidden layer in the feed-forward layers for the transformer blocks.
pub const FF_HIDDEN_DIMENSION: usize = 32;

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
