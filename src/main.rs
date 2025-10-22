mod data;
mod errors;
mod layers;
mod model;
mod params;
mod tokenizer;

use clap::Parser;
use ndarray::{Array1, Array2, Array3, Axis};

use crate::{
    layers::{
        Layer, embedding::EmbeddingLayer, multi_head_attention::MultiHeadAttentionLayer,
        normalization::Softmax,
    },
    params::{BATCH_SIZE, EMBED_DIMENSION, SEQUENCE_LENGTH},
    tokenizer::ByteTokenizer,
};

#[derive(Parser)]
struct CLIArgs {
    // #[arg(short, long, default_value = "data/moby_dick.txt")]
    #[arg(short, long, default_value = "data/moby_dick.txt")]
    data_file: Vec<String>,
}

fn main() {
    let args = CLIArgs::parse();

    println!("Using data file: {}", args.data_file.join(", "));
    let mut data =
        data::DataLoader::new(args.data_file, BATCH_SIZE, None).expect("Failed to load data");

    let tokenizer = ByteTokenizer;
    let vocab_size = tokenizer.vocab_size();

    let seed = None;
    let model = model::CrabformerModel::new(vocab_size, seed).expect("Failed to create model");

    let batch = data.next_batch().expect("no data").expect("batch is None");
    // let res = model.forward_batch(&batch);
    // let next_tokens = model.next_token_batch(&batch);

    // println!("Model output: {:?}", next_tokens);

    let mut output = model.forward_batch(&batch);

    output.softmax(2, None);

    let loss = batch_loss(&output, &batch.y);
    println!("Loss: {}", loss);

    // println!("Model output: {:?}", output);

    // for i in 0..next_tokens.len() {
    //     let predicted = tokenizer.decode(&vec![next_tokens[i]]);
    //     let input = tokenizer.decode(&batch.x.row(i).to_vec());

    //     let last_index = batch.y.dim().1 - 1;
    //     let actual = tokenizer.decode(&vec![batch.y.get((i, last_index)).cloned().unwrap()]);

    //     println!("Input sequence: {:?}", input);
    //     println!("Predicted: {}, Actual: {}", predicted, actual);
    // }

    // let embed_dim = 10;

    // let token_embedding_layer = EmbeddingLayer::new(vocab_size, EMBED_DIMENSION, None);
    // let position_embedding_layer = EmbeddingLayer::new(SEQUENCE_LENGTH, EMBED_DIMENSION, seed);

    // let positions = Array2::from_shape_fn((BATCH_SIZE, SEQUENCE_LENGTH), |(_, j)| {
    //     j as u32 // Each position in the sequence gets its index
    // });

    // let mut token_embedding_output = token_embedding_layer.forward(&batch.x);
    // let position_embedding_output = position_embedding_layer.forward(&positions);

    // token_embedding_output += &position_embedding_output;

    // let attention_layer = MultiHeadAttentionLayer::new(EMBED_DIMENSION, EMBED_DIMENSION, 2, seed)
    //     .and_then(|l| Ok(l.with_casual_mask().with_qkv_bias()))
    //     .unwrap();

    // let attention_output = attention_layer.forward(&token_embedding_output);
    // println!("Attention output: {:?}", attention_output);
    // println!("Batch data: {:?}", batch.x);
}

/// Computes the cross-entropy loss between predictions and targets.
/// Note that the targets is a 2D array containing the real next token after each
/// token in our sequences.
fn batch_loss(predictions: &Array3<f32>, targets: &Array2<u32>) -> f32 {
    let mut total_loss = 0.0;
    let batch_size = predictions.dim().0;

    // Iterate over each target vector in the batch
    for (batch_i, target_row) in targets.axis_iter(Axis(0)).enumerate() {
        // Get the corresponding prediction row from same batch
        let pred_row = predictions.index_axis(Axis(0), batch_i);

        // Iterate over each token in the sequence
        for (j, &target_token) in target_row.iter().enumerate() {
            // Get the predicted probabilities for the j-th token
            let predicted_probs = pred_row.index_axis(Axis(0), j);

            // Fetch the predicted probability using the target token index
            let predicted_prob = predicted_probs[target_token as usize];
            println!(
                "  Token {}: target={}, predicted_prob={}",
                j, target_token, predicted_prob
            );
            // Accumulate the negative log likelihood
            total_loss += -predicted_prob.ln();
        }
    }

    // Return the average loss over the batch
    total_loss / batch_size as f32
}
