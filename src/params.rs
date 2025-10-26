use std::sync::atomic::{AtomicU64, Ordering};

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use rand::{SeedableRng, rngs::StdRng};

pub const CHECKPOINTS_DIR: &str = "checkpoints";

pub const NUM_EPOCHS: usize = 50;
pub const BATCH_SIZE: usize = 16;
pub const SAVE_EVERY_N_STEPS: usize = 100;
pub const TEMPERATURE: f32 = 0.9;

/// Dimension of the hidden layer in the feed-forward layers for the transformer blocks.
pub const FF_HIDDEN_DIMENSION: usize = 256;
pub const ATTENTION_HEADS: usize = 8;
pub const TRANSFORMER_BLOCKS: usize = 8;
/// The length of each training sequence (T tokens).
pub const SEQUENCE_LENGTH: usize = 128;
pub const EMBED_DIMENSION: usize = 128;

/// The size of the data ring buffer used for training. A larger size improves batch shuffling.
pub const RING_BUFFER_SIZE: usize = 16 * SEQUENCE_LENGTH * BATCH_SIZE;
/// The number of new tokens to add to the ring buffer after reading each batch.
pub const RING_BUFFER_ADVANCE: usize = SEQUENCE_LENGTH * BATCH_SIZE;

pub const QKV_BIAS: bool = true;
pub const DROPOUT_RATE: f32 = 0.05;

pub const LEARNING_RATE: f32 = 0.00035;
pub const WEIGHT_DECAY: f32 = 0.01;
pub const ADAMW_BETA1: f32 = 0.9;
pub const ADAMW_BETA2: f32 = 0.999;
pub const ADAMW_EPSILON: f32 = 1e-8;
pub const GRADIENT_CLIP_NORM: f32 = 1.0;

pub const SEED: Option<u64> = Some(42);

pub static GLOBAL_RNG: Lazy<Mutex<StdRng>> = Lazy::new(|| {
    let rng = match SEED {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_os_rng(),
    };
    Mutex::new(rng)
});
