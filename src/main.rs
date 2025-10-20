mod data;
mod errors;
mod layers;
mod params;

use clap::Parser;
use ndarray::{Array1, Array2};

use crate::{
    layers::{embedding::EmbeddingLayer, self_attention::SelfAttentionLayer},
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
    // let embed_dim = 10;
    let seed = None;

    let token_embedding_layer = EmbeddingLayer::new(vocab_size, EMBED_DIMENSION, None);
    let position_embedding_layer = EmbeddingLayer::new(SEQUENCE_LENGTH, EMBED_DIMENSION, seed);

    let batch = data.next_batch().expect("no data").expect("batch is None");
    let positions = Array2::from_shape_fn((BATCH_SIZE, SEQUENCE_LENGTH), |(_, j)| {
        j as u32 // Each position in the sequence gets its index
    });

    let mut token_embedding_output = token_embedding_layer.forward(&batch.x);
    let position_embedding_output = position_embedding_layer.forward(&positions);
    println!("amogus");
    println!(
        "Token embedding output shape: {:?}",
        token_embedding_output.dim()
    );
    println!(
        "Position embedding output shape: {:?}",
        position_embedding_output.dim()
    );

    token_embedding_output += &position_embedding_output;

    let attention_layer = SelfAttentionLayer::new(EMBED_DIMENSION, seed).with_casual_mask();

    let rng = &mut params::get_rng(None);

    let attention_output = attention_layer.forward(&token_embedding_output, rng);
    println!("Attention output shape: {:?}", attention_output.dim());
    // println!("Batch data: {:?}", batch.x);
}
