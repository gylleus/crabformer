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

// impl Dropout for Tensor {
//     fn apply_dropout(&mut self, dropout_rate: f32, rng: &mut StdRng) {
//         match self {
//             Tensor::Array3(arr) => arr.apply_dropout(dropout_rate, rng),
//             Tensor::Array2(arr) => arr.apply_dropout(dropout_rate, rng),
//         }
//     }
// }
