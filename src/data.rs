use ndarray::Array2;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use crate::{
    errors::DataError,
    params::{RING_BUFFER_SIZE, SEQUENCE_LENGTH, get_rng},
};

pub struct Batch {
    pub x: Array2<u32>, // [batch_size, sequence_length]
    pub y: Array2<u32>, // [batch_size, sequence_length]
}

pub struct DataLoader {
    shuffler: RingShuffler<TokenStream>,
    batch_size: usize,
}

impl DataLoader {
    pub fn new(
        files: Vec<String>,
        batch_size: usize,
        seed: Option<u64>,
    ) -> Result<Self, DataError> {
        let token_stream = TokenStream::new(files);

        let mut shuffler = RingShuffler::new(
            token_stream,
            RING_BUFFER_SIZE,
            SEQUENCE_LENGTH,
            SEQUENCE_LENGTH,
            seed,
        );
        shuffler.warmup()?;
        Ok(Self {
            shuffler,
            batch_size,
        })
    }

    /// Produce the next batch. `Ok(None)` means the underlying stream hit EOF.
    pub fn next_batch(&mut self) -> Result<Option<Batch>, DataError> {
        self.shuffler.next_batch(self.batch_size)
    }
}

/// Keeps a circular buffer of size `cap` (S tokens).
/// At each batch, samples random **contiguous** windows inside the ring,
/// then advances the ring by `advance` fresh tokens from the stream.
struct RingShuffler<S>
where
    S: Iterator<Item = Result<u32, DataError>>,
{
    ring: Vec<u32>,
    cap: usize,             // S
    sequence_length: usize, // sequence length T

    head: usize, // next write index (oldest element lives at `head`)
    filled: bool,
    advance: usize, // how many new tokens to pull per batch (e.g., T)
    rng: StdRng,
    stream: S,
    exhausted: bool,
}

impl<S> RingShuffler<S>
where
    S: Iterator<Item = Result<u32, DataError>>,
{
    fn new(stream: S, cap: usize, t: usize, advance: usize, seed: Option<u64>) -> Self {
        assert!(cap >= t + 1, "ring capacity must be >= T+1");
        Self {
            ring: vec![0; cap],
            cap,
            head: 0,
            filled: false,
            sequence_length: t,
            advance: advance.max(1),
            rng: get_rng(seed),
            stream,
            exhausted: false,
        }
    }

    /// Fill the ring once with `cap` tokens in-order.
    pub fn warmup(&mut self) -> Result<(), DataError> {
        for _ in 0..self.cap {
            let tok = self.next_token()?;
            self.ring[self.head] = tok;
            self.head = (self.head + 1) % self.cap;
        }
        self.filled = true;
        Ok(())
    }

    /// Produce the next batch. `Ok(None)` means the underlying stream hit EOF.
    fn next_batch(&mut self, batch_size: usize) -> Result<Option<Batch>, DataError> {
        if !self.filled {
            return Err(DataError::Io("call warmup() before next_batch".into()));
        }
        if self.exhausted {
            return Ok(None);
        }

        // let mut x = vec![0u32; batch_size * self.sequence_length];
        // let mut y = vec![0u32; batch_size * self.sequence_length];
        let mut x = Array2::<u32>::zeros((batch_size, self.sequence_length));
        let mut y = Array2::<u32>::zeros((batch_size, self.sequence_length));

        // sample B random contiguous windows inside the ring
        let max_start = self.cap - (self.sequence_length + 1);
        for bi in 0..batch_size {
            let start = self.rng.random_range(0..=max_start);
            for k in 0..self.sequence_length {
                x[[bi, k]] = self.ring_get(start + k);
                y[[bi, k]] = self.ring_get(start + k + 1);
            }
        }

        // advance ring with fresh tokens
        for _ in 0..self.advance {
            match self.stream.next() {
                Some(Ok(tok)) => {
                    self.ring[self.head] = tok;
                    self.head = (self.head + 1) % self.cap;
                }
                Some(Err(e)) => return Err(e),
                None => {
                    // EOF: mark exhausted; still return the batch we just built
                    self.exhausted = true;
                    break;
                }
            }
        }

        Ok(Some(Batch { x, y }))
    }

    #[inline]
    fn ring_get(&self, logical: usize) -> u32 {
        // Oldest element is at `head`; logical index 0 means oldest.
        let idx = (self.head + logical) % self.cap;
        self.ring[idx]
    }

    #[inline]
    fn next_token(&mut self) -> Result<u32, DataError> {
        match self.stream.next() {
            Some(Ok(t)) => Ok(t),
            Some(Err(e)) => Err(e),
            None => Err(DataError::Io("unexpected EOF during warmup".into())),
        }
    }
}

struct TokenStream {
    files: Vec<String>,
    current_file_buffer: Option<BufReader<File>>,
    file_index: usize,
}

impl TokenStream {
    fn new(files: Vec<String>) -> Self {
        TokenStream {
            files,
            current_file_buffer: None,
            file_index: 0,
        }
    }

    fn open_next_file(&mut self) -> Result<bool, DataError> {
        let Some(next_file) = self.files.get(self.file_index) else {
            return Ok(false);
        };
        self.current_file_buffer = Some(BufReader::new(
            File::open(next_file).map_err(|e| DataError::FileError(e.to_string()))?,
        ));
        self.file_index += 1;
        Ok(true)
    }
}

impl Iterator for TokenStream {
    type Item = Result<u32, DataError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current_file_buffer.is_none() && self.open_next_file().ok()? == false {
                return None; // No more files to read
            }

            // self.current_file_buffer.as_mut().ok_or(DataError::Io(()))

            let Some(reader) = self.current_file_buffer.as_mut() else {
                return Some(Err(DataError::Io("Failed to get file buffer".into())));
            };

            let buf = match reader.fill_buf() {
                Ok(buf) if buf.is_empty() => {
                    self.current_file_buffer = None; // End of file reached
                    continue; // Move to the next file
                }
                Ok(buf) => buf,
                Err(e) => return Some(Err(DataError::Io(e.to_string()))),
            };
            let next = buf[0];
            reader.consume(1);

            // Return the byte as u32 token (simple byte-level tokenization)
            return Some(Ok(next as u32));
        }
    }
}
