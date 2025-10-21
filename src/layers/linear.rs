use ndarray::{Array1, Array2, Array3, ArrayBase, Data, Ix2};

use crate::{
    layers::{Layer, activation::GELU, xavier_initialized_array},
    params::get_rng,
};

#[derive(Debug, Clone)]
pub struct LinearLayer {
    pub weights: Array2<f32>,
    pub bias: Option<Array1<f32>>,
}

impl LinearLayer {
    pub fn new(dim_in: usize, dim_out: usize, seed: Option<u64>) -> Self {
        let mut rng = get_rng(seed);
        let weights = xavier_initialized_array(dim_in, dim_out, &mut rng);

        Self {
            weights,
            bias: None,
        }
    }

    pub fn with_bias(mut self) -> Self {
        let out_dim = self.weights.dim().1;
        self.bias = Some(Array1::zeros(out_dim));
        self
    }

    /// Forward pass for the linear layer (2D input).
    pub fn forward_2d<S>(&self, input: &ArrayBase<S, Ix2>) -> Array2<f32>
    where
        S: Data<Elem = f32>,
    {
        let mut output = input.dot(&self.weights);
        if let Some(bias) = &self.bias {
            output = output + bias;
        }
        output
    }

    /// Forward pass for the linear layer (3D input).
    /// Applies linear transformation on the last dimension.
    pub fn forward_3d(&self, input: &Array3<f32>) -> Array3<f32> {
        let (batch_size, seq_len, dim_in) = input.dim();
        let dim_out = self.weights.dim().1;

        // Reshape to 2D: (batch_size * seq_len, dim_in)
        let input_2d = input.to_shape((batch_size * seq_len, dim_in)).unwrap();

        // Apply linear transformation
        let output_2d = self.forward_2d(&input_2d);

        // Reshape back to 3D: (batch_size, seq_len, dim_out)
        output_2d
            .to_shape((batch_size, seq_len, dim_out))
            .unwrap()
            .to_owned()
    }
}

pub struct FeedForwardLayer {
    pub linear1: LinearLayer,
    pub linear2: LinearLayer,
}

impl FeedForwardLayer {
    pub fn new(dim_model: usize, dim_ff: usize, seed: Option<u64>) -> Self {
        let linear1 = LinearLayer::new(dim_model, dim_ff, seed);
        let linear2 = LinearLayer::new(dim_ff, dim_model, seed);

        Self { linear1, linear2 }
    }

    /// Forward pass for the feed-forward layer (2D input).
    /// input shape: (batch_size, dim_model)
    ///
    /// output shape: (batch_size, dim_model)
    pub fn forward<S>(&self, input: &ArrayBase<S, Ix2>) -> Array2<f32>
    where
        S: Data<Elem = f32>,
    {
        let mut hidden = self.linear1.forward_2d(input);
        // Apply non-linearity (GELU)
        hidden.apply_gelu();
        let output = self.linear2.forward_2d(&hidden);
        output
    }

    /// Forward pass for the feed-forward layer (3D input).
    /// input shape: (batch_size, seq_len, dim_model)
    ///
    /// output shape: (batch_size, seq_len, dim_model)
    pub fn forward_3d(&self, input: &Array3<f32>) -> Array3<f32> {
        let mut hidden = self.linear1.forward_3d(input);
        // Apply non-linearity (GELU)
        hidden.apply_gelu();
        let output = self.linear2.forward_3d(&hidden);
        output
    }
}
