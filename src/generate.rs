use std::io::{self, Write};

use ndarray::Array2;

use crate::{
    data::Batch,
    model::CrabformerModel,
    params::SEQUENCE_LENGTH,
    tokenizer::ByteTokenizer,
};

#[derive(Debug, Clone, Copy)]
pub enum SamplingStrategy {
    /// Sample from the full probability distribution
    Temperature(f32),
    /// Sample from the top-k most likely tokens
    TopK { k: usize, temperature: f32 },
    /// Greedy decoding - always pick the most likely token
    Greedy,
}

pub fn chat(model: &CrabformerModel) {
    let tokenizer = ByteTokenizer;

    println!("\n=== Crabformer Chat Interface ===");
    println!("Type your text and the model will continue it.");
    println!("Commands:");
    println!("  'quit' or 'exit' - Exit the chat");
    println!("  'greedy' - Use greedy decoding (most likely token)");
    println!("  'topk' - Use top-k sampling (k=50)");
    println!("  'temp' - Use temperature sampling\n");

    let mut sampling_strategy = SamplingStrategy::Greedy;

    loop {
        // Prompt user for input
        print!("\nYou: ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read input");

        let input = input.trim();

        // Check for exit commands
        if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit") {
            println!("Goodbye!");
            break;
        }

        // Check for strategy commands
        if input.eq_ignore_ascii_case("greedy") {
            sampling_strategy = SamplingStrategy::Greedy;
            println!("Switched to greedy decoding");
            continue;
        }
        if input.eq_ignore_ascii_case("topk") {
            sampling_strategy = SamplingStrategy::TopK {
                k: 50,
                temperature: 1.0,
            };
            println!("Switched to top-k sampling (k=50)");
            continue;
        }
        if input.eq_ignore_ascii_case("temp") {
            sampling_strategy = SamplingStrategy::Temperature(0.8);
            println!("Switched to temperature sampling (temp=0.8)");
            continue;
        }

        if input.is_empty() {
            continue;
        }

        // Generate continuation
        println!("Using strategy: {:?}", sampling_strategy);
        let continuation = generate_text(model, &tokenizer, input, 200, sampling_strategy);

        println!("\nModel: {}\n", continuation);
    }
}

fn generate_text(
    model: &CrabformerModel,
    tokenizer: &ByteTokenizer,
    prompt: &str,
    max_new_tokens: usize,
    sampling_strategy: SamplingStrategy,
) -> String {
    // Encode the prompt
    let mut tokens = tokenizer.encode(prompt);

    println!("Initial tokens: {:?}", tokens);
    println!("Generating {} new tokens...", max_new_tokens);

    // Generate tokens one at a time
    for i in 0..max_new_tokens {
        // Take the last SEQUENCE_LENGTH tokens (or fewer if we don't have that many yet)
        let context_len = tokens.len().min(SEQUENCE_LENGTH);
        let context_start = tokens.len().saturating_sub(SEQUENCE_LENGTH);
        let context = &tokens[context_start..];

        // Create padded context - pad at the START, not the end
        // This way the actual tokens are at the end of the sequence where the model predicts
        let mut padded_context = vec![0u32; SEQUENCE_LENGTH];
        let padding_len = SEQUENCE_LENGTH - context_len;
        padded_context[padding_len..].copy_from_slice(context);

        // Create batch with single sequence
        let x = Array2::from_shape_vec((1, SEQUENCE_LENGTH), padded_context)
            .expect("Failed to create input array");

        // We don't need y for generation, but Batch requires it
        let y = Array2::zeros((1, SEQUENCE_LENGTH));

        let batch = Batch { x, y };

        // Get model output (logits)
        let model_output = model.forward_batch(&batch);

        // Get logits for the last position in the sequence
        let last_token_logits = model_output.slice(ndarray::s![0, SEQUENCE_LENGTH - 1, ..]).to_owned();

        // Sample next token based on strategy
        let next_token = sample_token(&last_token_logits, sampling_strategy);

        // Print first few generated tokens for debugging
        if i < 10 {
            let c = if next_token < 128 && (next_token as u8).is_ascii_graphic() {
                next_token as u8 as char
            } else {
                '?'
            };
            println!("  Token {}: {} ('{}')", i, next_token, c);
        }

        // Stop if we generate certain control characters
        // But allow newlines (10) and other common whitespace
        if next_token == 0 {
            println!("Stopping: null byte generated");
            break;
        }

        tokens.push(next_token);

        // Also stop if we hit a newline for a cleaner output
        if next_token == 10 {  // newline
            break;
        }
    }

    // Decode all tokens
    tokenizer.decode(&tokens)
}

fn sample_token(logits: &ndarray::Array1<f32>, strategy: SamplingStrategy) -> u32 {
    use crate::layers::normalization::Softmax;
    use crate::params::GLOBAL_RNG;
    use rand::distr::{Distribution, weighted::WeightedIndex};

    match strategy {
        SamplingStrategy::Greedy => {
            // Just pick the argmax
            logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0)
        }
        SamplingStrategy::Temperature(temp) => {
            // Apply temperature and softmax, then sample
            let mut scaled_logits = logits.clone();
            scaled_logits.mapv_inplace(|x| x / temp);
            scaled_logits.softmax(0, None);

            let mut rng = GLOBAL_RNG.lock();
            let dist = WeightedIndex::new(&scaled_logits).unwrap();
            dist.sample(&mut *rng) as u32
        }
        SamplingStrategy::TopK { k, temperature } => {
            // Get top-k indices
            let mut indexed_logits: Vec<(usize, f32)> =
                logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Take top k
            indexed_logits.truncate(k);

            // Apply temperature to top-k logits
            let mut top_k_logits: Vec<f32> = indexed_logits
                .iter()
                .map(|(_, logit)| logit / temperature)
                .collect();

            // Convert to array and apply softmax
            let mut logits_array = ndarray::Array1::from_vec(top_k_logits);
            logits_array.softmax(0, None);

            // Sample from top-k
            let mut rng = GLOBAL_RNG.lock();
            let dist = WeightedIndex::new(&logits_array).unwrap();
            let sampled_idx = dist.sample(&mut *rng);

            indexed_logits[sampled_idx].0 as u32
        }
    }
}
