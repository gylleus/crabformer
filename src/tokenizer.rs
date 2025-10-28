pub struct ByteTokenizer;

impl ByteTokenizer {
    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.bytes().map(|b| b as u32).collect()
    }

    pub fn decode(&self, tokens: &[u32]) -> String {
        let bytes: Vec<u8> = tokens.iter().map(|&t| t as u8).collect();
        String::from_utf8_lossy(&bytes).to_string()
    }

    pub fn vocab_size(&self) -> usize {
        256 // 256 possible tokens (0-255)
    }
}
