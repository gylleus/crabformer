use parking_lot::RwLock;

use ndarray::{Array, Array2, Array3, Data};
use rand::{Rng, RngCore, TryRngCore, rngs::StdRng};
use serde::{Serialize, de::DeserializeOwned};

use crate::{
    adamw::ParamHandle, errors::ModelError, metrics::TrainingMetricsHandle, params::GLOBAL_RNG,
};

pub mod activation;
pub mod dropout;
pub mod embedding;
pub mod linear;
pub mod multi_head_attention;
pub mod normalization;
pub mod self_attention;
pub mod transformer_block;

/// Initialize an array with Xavier/Glorot initialization.
/// This should keep variance of activations and gradients roughly the same across layers, to keep training more stable.
pub fn xavier_initialized_array(fan_in: usize, fan_out: usize) -> Array2<f32> {
    let limit = (6.0 / (fan_in as f32 + fan_out as f32)).sqrt();

    Array::from_shape_fn((fan_in, fan_out), |_| {
        GLOBAL_RNG.lock().random_range(-limit..limit)
    })
}

/// Type alias for layers that take 3D f32 arrays as input and output
pub type Layer3Df32 = dyn Layer<Input = Array3<f32>, Output = Array3<f32>>;

pub trait Layer: ZeroGrad + Serialize + DeserializeOwned {
    type Input;
    type Output;

    fn name(&self) -> &str;

    fn forward(&self, input: &Self::Input) -> Self::Output;

    /// Backward pass: computes gradients with respect to inputs.
    ///
    /// # Arguments
    /// * `grad_output` - Gradient of loss with respect to this layer's output
    ///
    /// # Returns
    /// Gradient of loss with respect to this layer's input, or an error if backward
    /// is called without a prior forward_train call
    fn backward(&mut self, grad_output: &Self::Output) -> Result<Self::Input, ModelError>;

    /// Set the layer to training mode (enables caching for backward pass)
    fn set_train(&mut self, metrics_handle: TrainingMetricsHandle) {}

    /// Set the layer to evaluation mode (disables caching to save memory)
    fn set_eval(&mut self) {}

    fn get_params(&mut self) -> Vec<ParamHandle> {
        Vec::new()
    }
}

pub trait ZeroGrad {
    fn zero_grad(&mut self);
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ParamKey(pub u64);

/// Helper struct for caching layer parameters during training without requiring mutable references to the layer.

pub struct LayerCacheParam<T> {
    pub data: RwLock<Option<T>>,
    pub id: ParamKey,
    name: String,
}

/// This default implementation allows deserializing LayerCacheParam in layers
impl<T> Default for LayerCacheParam<T> {
    fn default() -> Self {
        Self::new("None".into())
    }
}

impl<T> LayerCacheParam<T> {
    pub fn new(name: String) -> Self {
        let id = rand::rng().next_u64();
        Self {
            data: RwLock::new(None),
            id: ParamKey(id),
            name,
        }
    }

    pub fn mut_ref(&self) -> parking_lot::RwLockWriteGuard<'_, Option<T>> {
        self.data.write()
    }

    pub fn read_ref(&self) -> Result<parking_lot::MappedRwLockReadGuard<'_, T>, ModelError> {
        let guard = self.data.read();

        // Check if the Option contains a value
        if guard.is_none() {
            return Err(ModelError::EmptyCache {
                name: self.name.clone(),
            });
        }

        // Map the guard to unwrap the Option while keeping the lock
        Ok(parking_lot::RwLockReadGuard::map(guard, |opt| {
            opt.as_ref().unwrap()
        }))
    }

    pub fn id(&self) -> ParamKey {
        self.id.clone()
    }

    pub fn clear(&self) {
        let mut data_lock = self.data.write();
        *data_lock = None;
    }
}
