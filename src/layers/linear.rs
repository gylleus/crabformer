use ndarray::{Array1, Array2, Array3, ArrayBase, Data, Ix2};
use serde::{Deserialize, Serialize};

use crate::{
    adamw,
    errors::ModelError,
    layers::{
        Layer, LayerCacheParam, ZeroGrad,
        activation::{GELU, gelu_derivative},
        xavier_initialized_array,
    },
};

#[derive(Serialize, Deserialize)]
/// Wrapper struct around 2D linear layer to provide 3D data interface
pub struct LinearLayer {
    // linear: LinearLayer2D,
    pub weights: Array2<f32>,
    pub bias: Option<Array1<f32>>,

    #[serde(skip)]
    weight_grad: LayerCacheParam<Array2<f32>>,
    #[serde(skip)]
    bias_grad: LayerCacheParam<Array1<f32>>,
    #[serde(skip)]
    last_input: LayerCacheParam<Array3<f32>>,

    training: bool,
    name: String,
}

impl LinearLayer {
    pub fn new(dim_in: usize, dim_out: usize, name: Option<String>) -> Self {
        // let linear_2d = LinearLayer2D::new(dim_in, dim_out, seed);
        let name = name.unwrap_or("LinearLayer".into());
        Self {
            // linear: linear_2d,
            // last_input_shape: LayerCacheParam::new(None),
            weights: xavier_initialized_array(dim_in, dim_out),
            bias: None,
            weight_grad: LayerCacheParam::new(format!("{}::weight_grad", name)),
            bias_grad: LayerCacheParam::new(format!("{}::bias_grad", name)),
            last_input: LayerCacheParam::new(format!("{}::last_input", name)),
            training: false,
            name,
        }
    }

    pub fn with_bias(mut self) -> Self {
        self.bias = Some(Array1::zeros(self.weights.dim().1));
        self
    }
}

impl Layer for LinearLayer {
    type Input = Array3<f32>;
    type Output = Array3<f32>;

    fn name(&self) -> &str {
        &self.name
    }

    fn forward(&self, input: &Self::Input) -> Self::Output {
        let (batch_size, seq_len, dim_in) = input.dim();
        let dim_out = self.weights.dim().1;

        // Cache the input shape for backward pass
        if self.training {
            *self.last_input.mut_ref() = Some(input.clone());
        }

        let input_2d = input.to_shape((batch_size * seq_len, dim_in)).unwrap();

        let mut output_2d = input_2d.dot(&self.weights);
        if let Some(bias) = &self.bias {
            output_2d = output_2d + bias;
        }

        // Reshape back to 3D: (batch_size, seq_len, dim_out)
        output_2d
            .to_shape((batch_size, seq_len, dim_out))
            .unwrap()
            .to_owned()
    }

    fn backward(&mut self, grad_output: &Self::Output) -> Result<Self::Input, ModelError> {
        // Get the cached input
        let input = self.last_input.read_ref()?;

        let (batch_size, seq_len, dim_in) = input.dim();
        let dim_out = self.weights.dim().1;

        let mut weight_grad_mut = self.weight_grad.mut_ref();
        let mut bias_grad_mut = self.bias_grad.mut_ref();

        // Lazy initialization of gradients on first backward pass
        if weight_grad_mut.is_none() {
            *weight_grad_mut = Some(Array2::zeros(self.weights.dim()));
        }
        if self.bias.is_some() && bias_grad_mut.is_none() {
            *bias_grad_mut = Some(Array1::zeros(dim_out));
        }

        // Reshape to 2D for computation
        let input_2d = input.to_shape((batch_size * seq_len, dim_in)).unwrap();
        let grad_output_2d = grad_output
            .to_shape((batch_size * seq_len, dim_out))
            .unwrap();

        // Gradient w.r.t. weights: dL/dW = input^T @ grad_output
        // Shape: (dim_in, batch*seq) @ (batch*seq, dim_out) = (dim_in, dim_out)
        let weight_grad = input_2d.t().dot(&grad_output_2d);

        // Accumulate gradients (important for batching)
        if let Some(wg) = weight_grad_mut.as_mut() {
            *wg = &*wg + &weight_grad;
        }

        // Gradient w.r.t. bias: sum over batch and sequence dimensions
        if let Some(bg) = bias_grad_mut.as_mut() {
            let bias_grad = grad_output_2d.sum_axis(ndarray::Axis(0));
            *bg = &*bg + &bias_grad;
        }

        // Gradient w.r.t. input: grad_output @ weights^T
        // Shape: (batch*seq, dim_out) @ (dim_out, dim_in) = (batch*seq, dim_in)
        let grad_input_2d = grad_output_2d.dot(&self.weights.t());

        // Reshape back to 3D
        Ok(grad_input_2d
            .to_shape((batch_size, seq_len, dim_in))
            .unwrap()
            .to_owned())
    }

    fn set_train(&mut self) {
        self.training = true;
    }

    fn set_eval(&mut self) {
        self.training = false;
    }

    fn get_params(&mut self) -> Vec<adamw::ParamHandle> {
        let mut params = vec![adamw::ParamHandle::Array2 {
            key: self.weight_grad.id(),
            data: &mut self.weights,
            grad: &self.weight_grad,
        }];

        // Add bias parameter if it exists
        if let Some(ref mut bias) = self.bias {
            params.push(adamw::ParamHandle::Array1 {
                key: self.bias_grad.id(),
                data: bias,
                grad: &self.bias_grad,
            });
        }

        params
    }
}

impl ZeroGrad for LinearLayer {
    fn zero_grad(&mut self) {
        if let Some(wg) = self.weight_grad.mut_ref().as_mut() {
            wg.fill(0.0);
        }
        if let Some(bg) = self.bias_grad.mut_ref().as_mut() {
            bg.fill(0.0);
        }
    }
}

#[derive(Serialize, Deserialize)]
/// Feed-Forward Layer with two linear transformations and GELU activation
pub struct FeedForwardLayer {
    pub linear1: LinearLayer,
    pub linear2: LinearLayer,

    #[serde(skip)]
    // Cache for backward pass (stores hidden state before GELU activation)
    last_hidden: LayerCacheParam<Array3<f32>>,
    training: bool,

    name: String,
}

impl FeedForwardLayer {
    pub fn new(dim_model: usize, dim_ff: usize, name: Option<String>) -> Self {
        let name = name.unwrap_or("FeedForwardLayer".into());

        let linear1 = LinearLayer::new(dim_model, dim_ff, Some(format!("{}::linear1", name)));
        let linear2 = LinearLayer::new(dim_ff, dim_model, Some(format!("{}::linear2", name)));

        Self {
            linear1,
            linear2,
            last_hidden: LayerCacheParam::new(format!("{}::last_hidden", name)),
            training: false,
            name,
        }
    }
}

impl Layer for FeedForwardLayer {
    type Input = Array3<f32>;
    type Output = Array3<f32>;

    fn name(&self) -> &str {
        &self.name
    }

    fn forward(&self, input: &Self::Input) -> Self::Output {
        let mut hidden = self.linear1.forward(input);

        // Cache the hidden state BEFORE activation for backward pass
        if self.training {
            // We need to cache before GELU for the derivative
            *self.last_hidden.mut_ref() = Some(hidden.clone());
        }

        // Apply non-linearity (GELU)
        hidden.apply_gelu();
        let output = self.linear2.forward(&hidden);
        output
    }

    fn backward(&mut self, grad_output: &Self::Output) -> Result<Self::Input, ModelError> {
        let hidden = self.last_hidden.read_ref()?;

        // Backprop through second linear layer
        let mut grad_hidden = self.linear2.backward(grad_output)?;

        // Backprop through GELU activation
        // GELU'(x) needs the original input to GELU (which is the hidden state)
        grad_hidden = grad_hidden * &hidden.mapv(|x| gelu_derivative(x));

        // Backprop through first linear layer
        let grad_input = self.linear1.backward(&grad_hidden)?;

        Ok(grad_input)
    }

    fn set_train(&mut self) {
        self.training = true;
        self.linear1.set_train();
        self.linear2.set_train();
    }

    fn set_eval(&mut self) {
        self.training = false;
        self.linear1.set_eval();
        self.linear2.set_eval();
    }

    fn get_params(&mut self) -> Vec<adamw::ParamHandle> {
        let mut params = Vec::new();

        // Collect parameters from both linear layers
        params.extend(self.linear1.get_params());
        params.extend(self.linear2.get_params());

        params
    }
}

impl ZeroGrad for FeedForwardLayer {
    fn zero_grad(&mut self) {
        self.linear1.zero_grad();
        self.linear2.zero_grad();
    }
}
