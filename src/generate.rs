use std::io::{self, Write};

use ndarray::{Array1, Array2};
use rand::distr::{Distribution, weighted::WeightedIndex};

use crate::{
    data::Batch, layers::normalization::Softmax, model::CrabformerModel, rng::GLOBAL_RNG,
    tokenizer::ByteTokenizer,
};

pub fn chat(model: &CrabformerModel, temperature: f32, top_k: usize) {
    let tokenizer = ByteTokenizer;

    println!("\n=== Crabformer Chat Interface ===");
    println!("Type your text and the model will continue it.");

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

        if input.is_empty() {
            continue;
        }

        generate_text(model, &tokenizer, input, 200, temperature, top_k);

        println!("\n---");
    }
}

fn generate_text(
    model: &CrabformerModel,
    tokenizer: &ByteTokenizer,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f32,
    top_k: usize,
) -> String {
    // Encode the prompt
    let mut tokens = tokenizer.encode(prompt);

    print!("Crabformer: ");
    io::stdout().flush().unwrap();

    // Generate tokens one at a time
    for _i in 0..max_new_tokens {
        // Take the last SEQUENCE_LENGTH tokens (or fewer if we don't have that many yet)
        let context_len = tokens.len().min(model.config.seq_length);
        let context_start = tokens.len().saturating_sub(model.config.seq_length);
        let context = &tokens[context_start..];

        // Create padded context - pad at the START, not the end
        // This way the actual tokens are at the end of the sequence where the model predicts
        let mut padded_context = vec![0u32; model.config.seq_length];
        let padding_len = model.config.seq_length - context_len;
        padded_context[padding_len..].copy_from_slice(context);

        // Create batch with single sequence
        let x = Array2::from_shape_vec((1, model.config.seq_length), padded_context)
            .expect("Failed to create input array");

        // We don't need y for generation, but Batch requires it
        let y = Array2::zeros((1, model.config.seq_length));

        let batch = Batch { x, y };

        // Get model output (logits)
        let model_output = model.forward_batch(&batch);

        // Get logits for the last position in the sequence
        let last_token_logits = model_output
            .slice(ndarray::s![0, model.config.seq_length - 1, ..])
            .to_owned();

        // Sample next token based on strategy
        let next_token = sample_token(&last_token_logits, temperature, top_k);
        tokens.push(next_token);

        print!("{}", tokenizer.decode(&[next_token]));
        io::stdout().flush().unwrap();
    }

    // Decode all tokens
    tokenizer.decode(&tokens)
}

fn sample_token(logits: &Array1<f32>, temperature: f32, top_k: usize) -> u32 {
    // Get top-k indices
    let mut indexed_logits: Vec<(usize, f32)> =
        logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Take top k
    indexed_logits.truncate(top_k);

    // Apply temperature to top-k logits
    let top_k_logits: Vec<f32> = indexed_logits
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
