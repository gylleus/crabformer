use std::collections::HashMap;

use ndarray::{Array1, Array2, ArrayD};

use crate::layers::{LayerCacheParam, ParamKey};

/// Represents a trainable parameter with its optimizer state.
/// The parameter can be of any dimensionality (1D for bias, 2D for weights, etc.)
pub struct AdamState {
    /// First moment estimate (exponentially decaying average of gradients)
    m: ArrayD<f32>,
    /// Second moment estimate (exponentially decaying average of squared gradients)
    v: ArrayD<f32>,
}

impl AdamState {
    /// Creates a new parameter with the given data.
    /// Initializes gradient and optimizer state to zeros with the same shape.
    pub fn new(shape: &[usize]) -> Self {
        Self {
            m: ArrayD::zeros(shape),
            v: ArrayD::zeros(shape),
        }
    }
}

pub trait Param {
    fn get_handle(&mut self) -> ParamHandle;
}

pub enum ParamHandle<'a> {
    Array1 {
        key: ParamKey,
        data: &'a mut Array1<f32>,
        grad: &'a LayerCacheParam<Array1<f32>>,
    },
    Array2 {
        key: ParamKey,
        data: &'a mut Array2<f32>,
        grad: &'a LayerCacheParam<Array2<f32>>,
    },
}

/// AdamW optimizer with decoupled weight decay.
///
/// AdamW improves upon Adam by decoupling the weight decay from the gradient-based update.
/// This leads to better generalization and is now considered best practice.
///
/// Paper: "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019)
pub struct AdamWOptimizer {
    /// Learning rate (step size)
    learning_rate: f32,
    /// Exponential decay rate for first moment estimates (typically 0.9)
    beta1: f32,
    /// Exponential decay rate for second moment estimates (typically 0.999)
    beta2: f32,
    /// Small constant for numerical stability (typically 1e-8)
    epsilon: f32,
    /// Weight decay coefficient for L2 regularization (typically 0.01)
    weight_decay: f32,
    /// Current time step (for bias correction)
    t: usize,
    /// Storage for all parameters
    parameters: HashMap<ParamKey, AdamState>,
}

impl AdamWOptimizer {
    /// Creates a new AdamW optimizer with the given hyperparameters.
    ///
    /// # Arguments
    /// * `learning_rate` - Step size (typical: 0.001 to 0.0001 for transformers)
    /// * `beta1` - Exponential decay for first moment (typical: 0.9)
    /// * `beta2` - Exponential decay for second moment (typical: 0.999)
    /// * `epsilon` - Numerical stability constant (typical: 1e-8)
    /// * `weight_decay` - Weight decay coefficient (typical: 0.01)
    pub fn new(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    ) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            t: 0,
            parameters: HashMap::new(),
        }
    }

    // /// Creates an AdamW optimizer with default hyperparameters commonly used for transformers.
    // pub fn default() -> Self {
    //     Self::new(
    //         0.0001, // learning_rate
    //         0.9,    // beta1
    //         0.999,  // beta2
    //         1e-8,   // epsilon
    //         0.01,   // weight_decay
    //     )
    // }

    /// Performs a single optimization step.
    /// Updates all registered parameters based on their accumulated gradients.
    pub fn step(&mut self, params: &mut [ParamHandle]) {
        self.t += 1;

        // Compute bias correction terms
        // These correct for the initialization bias (m and v start at zero)
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        for param in params.iter_mut() {
            match param {
                ParamHandle::Array1 { key, data, grad } => {
                    self.step_param(key, data, grad, bias_correction1, bias_correction2);
                }
                ParamHandle::Array2 { key, data, grad } => {
                    self.step_param(key, data, grad, bias_correction1, bias_correction2);
                }
            }
        }
    }

    fn step_param<D>(
        &mut self,
        key: &ParamKey,
        data: &mut ndarray::Array<f32, D>,
        grad_cache: &LayerCacheParam<ndarray::Array<f32, D>>,
        bias_correction1: f32,
        bias_correction2: f32,
    ) where
        D: ndarray::Dimension,
    {
        // Get the gradient from the LayerCacheParam
        let grad_guard = grad_cache.mut_ref();
        let grad = match grad_guard.as_ref() {
            Some(g) => g,
            None => return, // Skip if gradient not initialized
        };

        // Convert to dynamic arrays for storage and computation
        let grad_dyn = grad.view().into_dyn();
        let data_dyn = data.view().into_dyn();

        // Initialize optimizer state for this parameter if not seen before
        if !self.parameters.contains_key(key) {
            self.parameters
                .insert(key.clone(), AdamState::new(data_dyn.shape()));
        }
        let param_state = self.parameters.get_mut(key).unwrap();

        // Update biased first moment estimate: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        param_state.m = &param_state.m * self.beta1 + &grad_dyn * (1.0 - self.beta1);

        // Update biased second moment estimate: v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        param_state.v =
            &param_state.v * self.beta2 + &(&grad_dyn * &grad_dyn) * (1.0 - self.beta2);

        // Compute bias-corrected first moment: m_hat = m_t / (1 - beta1^t)
        let m_hat = &param_state.m / bias_correction1;

        // Compute bias-corrected second moment: v_hat = v_t / (1 - beta2^t)
        let v_hat = &param_state.v / bias_correction2;

        // AdamW update rule:
        // theta_t = theta_{t-1} - lr * (m_hat / (sqrt(v_hat) + epsilon) + weight_decay * theta_{t-1})
        // where weight_decay is the weight_decay coefficient
        //
        // This is different from Adam which applies weight decay through the gradient.
        // AdamW decouples weight decay from gradient-based optimization.

        let adaptive_lr = &m_hat / &(v_hat.mapv(|x| x.sqrt()) + self.epsilon);

        // Apply weight decay directly to parameters (AdamW's key innovation)
        let weight_decay_term = &data_dyn * self.weight_decay;

        // Compute the update
        let update = &data_dyn - &adaptive_lr * self.learning_rate - &weight_decay_term * self.learning_rate;

        // Convert back to original dimensionality and assign
        data.assign(&update.into_dimensionality::<D>().unwrap());
    }
}
