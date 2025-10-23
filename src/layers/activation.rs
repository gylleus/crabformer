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
        self.mapv_inplace(|x| gelu(x));
    }
}

/// GELU activation function: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
/// Approximation used in the original paper
pub fn gelu(x: f32) -> f32 {
    let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x.powi(3))).tanh())
}

/// Derivative of GELU activation function
/// This is used during backpropagation
pub fn gelu_derivative(x: f32) -> f32 {
    let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
    let tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x.powi(3));
    let tanh_val = tanh_arg.tanh();
    let sech2 = 1.0 - tanh_val.powi(2);

    0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * x.powi(2))
}
