use thiserror::Error;

#[derive(Debug, Error)]
pub enum DataError {
    #[error("I/O error: {0}")]
    Io(String),
    #[error("File error: {0}")]
    FileError(String),
}
