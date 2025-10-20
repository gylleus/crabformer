use rand::{SeedableRng, rngs::StdRng};

pub const BATCH_SIZE: usize = 16;

/// The length of each training sequence (T tokens).
pub const SEQUENCE_LENGTH: usize = 4;
/// The size of the data ring buffer used for training. A larger size improves batch shuffling.
pub const RING_BUFFER_SIZE: usize = SEQUENCE_LENGTH * 16;
/// The number of new tokens to add to the ring buffer after reading each batch.
pub const RING_BUFFER_ADVANCE: usize = SEQUENCE_LENGTH;

pub const EMBED_DIMENSION: usize = 8;

pub fn get_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_os_rng(),
    }
}
