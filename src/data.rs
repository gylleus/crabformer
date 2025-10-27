use ndarray::Array2;
use rand::Rng;
use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use crate::{errors::DataError, rng::GLOBAL_RNG};

pub struct Batch {
    pub x: Array2<u32>, // [batch_size, sequence_length]
    pub y: Array2<u32>, // [batch_size, sequence_length]
}

pub struct DataLoader {
    shuffler: RingShuffler<TokenStream>,
    total_batches: usize,
    files: Vec<String>,

    batch_size: usize,
    buffer_size: usize,
    sequence_length: usize,
    advance: usize,
}

impl DataLoader {
    pub fn new(
        files: Vec<String>,
        sequence_length: usize,
        batch_size: usize,
    ) -> Result<Self, DataError> {
        // Number of bytes to advance the ring buffer after each batch
        let advance = sequence_length * batch_size;
        let buffer_size = 16 * advance;

        let total_batches = count_total_batches(&files, advance, buffer_size)?;
        let total_tokens = count_total_tokens(&files)?;
        let token_stream = TokenStream::new(files.clone());

        let mut shuffler = RingShuffler::new(
            token_stream,
            buffer_size,
            sequence_length,
            advance,
            total_tokens,
        );
        shuffler.warmup()?;

        Ok(Self {
            shuffler,
            batch_size,
            buffer_size,
            sequence_length,
            advance,
            total_batches,
            files,
        })
    }

    /// Produce the next batch. `Ok(None)` means the underlying stream hit EOF.
    pub fn next_batch(&mut self) -> Result<Option<Batch>, DataError> {
        self.shuffler.next_batch(self.batch_size)
    }

    pub fn total_batches(&self) -> usize {
        self.total_batches
    }

    pub fn reset(&mut self) -> Result<(), DataError> {
        let total_tokens = count_total_tokens(&self.files)?;
        let token_stream = TokenStream::new(self.files.clone());

        let mut shuffler = RingShuffler::new(
            token_stream,
            self.buffer_size,
            self.sequence_length,
            self.advance,
            total_tokens,
        );
        shuffler.warmup()?;
        self.shuffler = shuffler;
        Ok(())
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

    stream: S,
    exhausted: bool,
    tokens_consumed: usize, // Total tokens read from stream
    max_tokens: usize,      // Stop after consuming this many tokens
}

impl<S> RingShuffler<S>
where
    S: Iterator<Item = Result<u32, DataError>>,
{
    fn new(stream: S, cap: usize, t: usize, advance: usize, max_tokens: usize) -> Self {
        assert!(cap > t, "ring capacity must be > T");
        Self {
            ring: vec![0; cap],
            cap,
            head: 0,
            filled: false,
            sequence_length: t,
            advance: advance.max(1),

            stream,
            exhausted: false,
            tokens_consumed: 0,
            max_tokens,
        }
    }

    /// Fill the ring once with `cap` tokens in-order.
    pub fn warmup(&mut self) -> Result<(), DataError> {
        for _ in 0..self.cap {
            let tok = self.next_token()?;
            self.ring[self.head] = tok;
            self.head = (self.head + 1) % self.cap;
            self.tokens_consumed += 1;
        }
        self.filled = true;
        Ok(())
    }

    /// Produce the next batch. `Ok(None)` means the underlying stream hit EOF or max tokens reached.
    fn next_batch(&mut self, batch_size: usize) -> Result<Option<Batch>, DataError> {
        if !self.filled {
            return Err(DataError::Io("call warmup() before next_batch".into()));
        }
        if self.exhausted || self.tokens_consumed >= self.max_tokens {
            return Ok(None);
        }

        let mut x = Array2::<u32>::zeros((batch_size, self.sequence_length));
        let mut y = Array2::<u32>::zeros((batch_size, self.sequence_length));

        // sample B random contiguous windows inside the ring
        let max_start = self.cap - (self.sequence_length + 1);
        let mut rng = GLOBAL_RNG.lock();

        for bi in 0..batch_size {
            let start = rng.random_range(0..=max_start);
            for k in 0..self.sequence_length {
                x[[bi, k]] = self.ring_get(start + k);
                y[[bi, k]] = self.ring_get(start + k + 1);
            }
        }

        for _ in 0..self.advance {
            // Stop if we've already consumed all tokens
            if self.tokens_consumed >= self.max_tokens {
                self.exhausted = true;
                break;
            }

            match self.stream.next() {
                Some(Ok(tok)) => {
                    self.ring[self.head] = tok;
                    self.head = (self.head + 1) % self.cap;
                    self.tokens_consumed += 1;
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
            if self.current_file_buffer.is_none() && !self.open_next_file().ok()? {
                return None; // No more files to read
            }

            let Some(reader) = self.current_file_buffer.as_mut() else {
                return Some(Err(DataError::Io("Failed to get file buffer".into())));
            };

            let buf = match reader.fill_buf() {
                Ok([]) => {
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

fn count_total_tokens(data_files: &Vec<String>) -> Result<usize, DataError> {
    let mut total_tokens = 0usize;

    for file_path in data_files {
        let file = File::open(file_path).map_err(|e| DataError::FileError(e.to_string()))?;
        let file_length = file
            .metadata()
            .map_err(|e| DataError::FileError(e.to_string()))?
            .len();

        total_tokens += file_length as usize;
    }

    Ok(total_tokens)
}

fn count_total_batches(
    data_files: &Vec<String>,
    advance: usize,
    buffer_size: usize,
) -> Result<usize, DataError> {
    let total_tokens = count_total_tokens(data_files)?;

    // With ring shuffler: we advance by `advance` tokens per batch
    // After warmup, remaining tokens divided by advance per batch gives us batch count
    let remaining_after_warmup = total_tokens.saturating_sub(buffer_size);
    let total_batches = remaining_after_warmup / advance;

    Ok(total_batches)
}
