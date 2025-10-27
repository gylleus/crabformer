use ndarray::{Array, RemoveAxis, parallel::prelude::IntoParallelRefMutIterator};
use rand::Rng;
use rayon::prelude::*;

use crate::rng::GLOBAL_RNG;

pub trait Dropout {
    /// Apply dropout during forward pass.
    /// Returns the dropout mask that should be saved for backward pass.
    fn apply_dropout(&mut self, dropout_rate: f32) -> Option<Vec<bool>>;

    /// Apply the same dropout mask during backward pass.
    fn apply_dropout_backward(&mut self, mask: &[bool], dropout_rate: f32);
}

impl<D> Dropout for Array<f32, D>
where
    D: ndarray::Dimension + RemoveAxis,
{
    fn apply_dropout(&mut self, dropout_rate: f32) -> Option<Vec<bool>> {
        if dropout_rate == 0.0 {
            return None;
        }
        let keep = 1.0 - dropout_rate;

        let len = self.len();
        // Create a flat view for efficient iteration
        let flat = self
            .as_slice_memory_order_mut()
            .expect("dropout expects contiguous array");

        // Draw mask with a single RNG lock.
        let mask: Vec<bool> = {
            let mut rng = GLOBAL_RNG.lock();
            (0..len)
                .map(|_| {
                    let r = rng.random::<f32>();
                    r >= dropout_rate
                })
                .collect()
        };

        flat.par_iter_mut()
            .zip(mask.par_iter())
            .for_each(|(elem, &keep_elem)| {
                if !keep_elem {
                    *elem = 0.0;
                } else {
                    *elem /= keep;
                }
            });

        Some(mask)
    }

    fn apply_dropout_backward(&mut self, mask: &[bool], dropout_rate: f32) {
        if dropout_rate == 0.0 {
            return;
        }
        let keep = 1.0 - dropout_rate;

        let flat = self
            .as_slice_memory_order_mut()
            .expect("dropout expects contiguous array");

        flat.par_iter_mut()
            .zip(mask.par_iter())
            .for_each(|(elem, &keep_elem)| {
                if !keep_elem {
                    *elem = 0.0;
                } else {
                    *elem /= keep;
                }
            });
    }
}
