pub mod activation;
pub mod dropout;
pub mod embedding;
pub mod feed_forward;
pub mod linear;
pub mod multi_head_attention;
pub mod normalization;
pub mod transformer_block;

use parking_lot::RwLock;

use ndarray::{Array, Array2};
use rand::{Rng, RngCore};
use serde::{Serialize, de::DeserializeOwned};

use crate::{
    adamw::ParamHandle, errors::ModelError, metrics::TrainingMetricsHandle, rng::GLOBAL_RNG,
};

/// Initialize an array with Xavier/Glorot initialization.
/// This aims to keep variance of activations and gradients roughly the same across layers, to keep training more stable.
pub fn xavier_initialized_array(fan_in: usize, fan_out: usize) -> Array2<f32> {
    let limit = (6.0 / (fan_in as f32 + fan_out as f32)).sqrt();

    Array::from_shape_fn((fan_in, fan_out), |_| {
        GLOBAL_RNG.lock().random_range(-limit..limit)
    })
}

/// Core trait for all layers of the model.
/// While the trait is not used directly in the model implementation, it provides a common interface for layer implementations.
pub trait Layer: ZeroGrad + Serialize + DeserializeOwned {
    type Input;
    type Output;

    #[allow(dead_code)]
    fn name(&self) -> &str;

    /// Forward pass: computes the layer output from the input.
    /// In the case of training, the layer caches intermediate results for the backward pass if needed.
    fn forward(&self, input: &Self::Input) -> Self::Output;

    /// Backward pass: computes and caches gradients with respect to inputs.
    fn backward(&mut self, grad_output: &Self::Output) -> Result<Self::Input, ModelError>;

    /// Set the layer to training mode (enables caching for backward pass)
    fn set_train(&mut self, _metrics_handle: TrainingMetricsHandle) {}

    /// Set the layer to evaluation mode (disables caching to save memory)
    fn set_eval(&mut self) {}

    /// Get mutable references to the layer parameters for optimization.
    fn get_params(&mut self) -> Vec<ParamHandle<'_>> {
        Vec::new()
    }
}

pub trait ZeroGrad {
    fn zero_grad(&mut self);
}

/// Unique identifier for layer parameters.
/// This is used by the optimizer to continuously track parameters.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ParamKey(pub u64);

/// Helper struct for caching layer parameters during training without requiring mutable references to the layer itself.
/// Using a lock like this adds a small overhead, but it is negligible compared to the rest of the model computations.
pub struct LayerCacheParam<T> {
    pub data: RwLock<Option<T>>,
    pub key: ParamKey,
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
            key: ParamKey(id),
            name,
        }
    }

    /// Returns a mutable reference to the cached data while holding the write lock.
    pub fn mut_ref(&self) -> parking_lot::RwLockWriteGuard<'_, Option<T>> {
        self.data.write()
    }

    /// Returns a read-only reference to the cached data while holding a read lock.
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

    pub fn key(&self) -> ParamKey {
        self.key.clone()
    }

    pub fn clear(&self) {
        let mut data_lock = self.data.write();
        *data_lock = None;
    }
}
