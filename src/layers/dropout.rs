use std::sync::atomic::{AtomicU64, Ordering};

use ndarray::{Array, Array3, RemoveAxis};
use rand::{Rng, rngs::StdRng};

use crate::{layers::Layer, params::MutableRng};

pub trait Dropout {
    fn apply_dropout(&mut self, dropout_rate: f32, rng: &mut StdRng);
}

impl<D> Dropout for Array<f32, D>
where
    D: ndarray::Dimension + RemoveAxis,
{
    fn apply_dropout(&mut self, dropout_rate: f32, rng: &mut StdRng) {
        for elem in self.iter_mut() {
            let noise = rng.random_range(0.0..1.0);
            if noise < dropout_rate {
                *elem = 0.0;
            } else {
                *elem = *elem / (1.0 - dropout_rate);
            }
        }
    }
}

// pub struct DropoutLayer {
//     pub dropout_rate: f32,
//     // Internal counter to increment the RNG state for each forward pass while keeping reproducibility.
//     rng: MutableRng,
// }

// impl Layer for DropoutLayer {
//     fn forward(&self, input: &Array3<f32>) -> Array3<f32> {
//         let mut output = input.clone();
//         output.with_dropout(self.dropout_rate, &mut self.rng.get_rng());
//         output
//     }
// }
