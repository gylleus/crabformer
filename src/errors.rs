use thiserror::Error;

#[derive(Debug, Error)]
pub enum DataError {
    #[error("I/O error: {0}")]
    Io(String),
    #[error("File error: {0}")]
    FileError(String),
}

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),
    #[error("{name}: layer parameter cache is empty")]
    EmptyCache { name: String },
    #[error("Data error: {0}")]
    DataError(#[from] DataError),
    #[error("Optimizer error: {0}")]
    OptimizerError(String),
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("I/O error: {0}")]
    IOError(String),
}
