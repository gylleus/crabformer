use ndarray::{Array, RemoveAxis, parallel::prelude::IntoParallelRefMutIterator};
use rand::{Rng, rngs::StdRng};
use rayon::prelude::*;

use crate::params::GLOBAL_RNG;

pub trait Dropout {
    fn apply_dropout(&mut self, dropout_rate: f32);
}

impl<D> Dropout for Array<f32, D>
where
    D: ndarray::Dimension + RemoveAxis,
{
    fn apply_dropout(&mut self, dropout_rate: f32) {
        if dropout_rate == 0.0 {
            return;
        }
        let keep = 1.0 - dropout_rate;

        let len = self.len();
        // Create a flat view for efficient iteration
        let flat = self
            .as_slice_memory_order_mut()
            .expect("dropout expects contiguous array");

        // Draw mask with a single RNG lock.
        let mask: Vec<f32> = {
            let mut rng = GLOBAL_RNG.lock();
            (0..len).map(|_| rng.random()).collect()
        };

        flat.par_iter_mut()
            .zip(mask.into_par_iter())
            .for_each(|(elem, noise)| {
                if noise < dropout_rate {
                    *elem = 0.0;
                } else {
                    *elem /= keep;
                }
            });
    }
}
