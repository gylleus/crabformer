use ndarray::{Array, RemoveAxis};

pub trait Normalized {
    fn softmax(&mut self, axis: usize);
}

impl<D> Normalized for Array<f32, D>
where
    D: ndarray::Dimension + RemoveAxis,
{
    fn softmax(&mut self, dim: usize) {
        for mut axis in self.axis_iter_mut(ndarray::Axis(dim)) {
            let local_max = axis.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            // Subtract by local max for numerical stability
            let sum_exp: f32 = axis.iter().map(|&v| (v - local_max).exp()).sum();
            for v in axis.iter_mut() {
                *v = (*v - local_max).exp() / sum_exp;
            }
        }
    }
}
