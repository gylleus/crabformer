mod data;
mod errors;
mod layers;
mod model;
mod params;

use clap::Parser;
use ndarray::{Array1, Array2};

use crate::{
    data::decode_bytes,
    layers::{Layer, embedding::EmbeddingLayer, multi_head_attention::MultiHeadAttentionLayer},
    params::{BATCH_SIZE, EMBED_DIMENSION, SEQUENCE_LENGTH},
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

    // let vocab_size = data.vocab_size();
    let vocab_size = data.vocab_size();

    let seed = None;
    let model = model::CrabformerModel::new(vocab_size, seed).expect("Failed to create model");

    let batch = data.next_batch().expect("no data").expect("batch is None");
    // let res = model.forward_batch(&batch);
    let next_tokens = model.next_token_batch(&batch);

    println!("Model output: {:?}", next_tokens);

    for i in 0..next_tokens.len() {
        let predicted = decode_bytes(&vec![next_tokens[i]]);
        let input = decode_bytes(&batch.x.row(i).to_vec());

        let last_index = batch.y.dim().1 - 1;
        let actual = decode_bytes(&vec![batch.y.get((i, last_index)).cloned().unwrap()]);

        println!("Input sequence: {:?}", input);
        println!("Predicted: {}, Actual: {}", predicted, actual);
    }

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
