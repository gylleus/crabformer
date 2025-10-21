use ndarray::{Array, RemoveAxis};

/// GELU (Gaussian Error Linear Unit) activation.
pub trait GELU {
    fn apply_gelu(&mut self);
}

impl<D> GELU for Array<f32, D>
where
    D: ndarray::Dimension + RemoveAxis,
{
    fn apply_gelu(&mut self) {
        self.mapv_inplace(|x| 0.5 * x * (1.0 + (x / std::f32::consts::SQRT_2).tanh()));
    }
}
