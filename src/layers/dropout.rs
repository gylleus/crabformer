use ndarray::{Array, RemoveAxis};
use rand::{Rng, rngs::StdRng};

pub trait Dropout {
    fn with_dropout(&mut self, dropout_rate: f32, rng: &mut StdRng);
}

impl<D> Dropout for Array<f32, D>
where
    D: ndarray::Dimension + RemoveAxis,
{
    fn with_dropout(&mut self, dropout_rate: f32, rng: &mut StdRng) {
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
