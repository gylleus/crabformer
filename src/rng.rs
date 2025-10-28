use once_cell::sync::Lazy;
use parking_lot::Mutex;
use rand::{SeedableRng, rngs::StdRng};

pub const SEED: Option<u64> = Some(42);

/// Global RNG instance protected by a Mutex for thread-safe access.
/// Uses the configured seed if set for reproducibility.
pub static GLOBAL_RNG: Lazy<Mutex<StdRng>> = Lazy::new(|| {
    let rng = match SEED {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_os_rng(),
    };
    Mutex::new(rng)
});
