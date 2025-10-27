mod adamw;
mod dashboard;
mod data;
mod errors;
mod generate;
mod layers;
mod loss;
mod metrics;
mod model;
mod rng;
mod tokenizer;

use clap::{Parser, Subcommand};

use crate::{model::ModelConfig, tokenizer::ByteTokenizer};

#[derive(Parser)]
struct CLIArgs {
    #[clap(subcommand)]
    training: SubCommands,
}

// pub const NUM_EPOCHS: usize = 200;
// pub const BATCH_SIZE: usize = 16;
// pub const SAVE_EVERY_N_STEPS: usize = 100;

// /// Dimension of the hidden layer in the feed-forward layers for the transformer blocks.
// pub const EMBED_DIMENSION: usize = 192 * 2;
// pub const FF_HIDDEN_DIMENSION: usize = 4 * EMBED_DIMENSION;
// pub const ATTENTION_HEADS: usize = 4;
// pub const TRANSFORMER_BLOCKS: usize = 4;
// /// The length of each training sequence (T tokens).
// pub const SEQUENCE_LENGTH: usize = 128;

// /// The size of the data ring buffer used for training. A larger size improves batch shuffling.
// pub const RING_BUFFER_SIZE: usize = 16 * SEQUENCE_LENGTH * BATCH_SIZE;
// pub const RING_BUFFER_ADVANCE: usize = SEQUENCE_LENGTH * BATCH_SIZE;

// pub const QKV_BIAS: bool = true;
// pub const DROPOUT_RATE: f32 = 0.01;

// pub const LEARNING_RATE: f32 = 5e-4;
// pub const WEIGHT_DECAY: f32 = 5e-3;

#[derive(Subcommand)]
enum SubCommands {
    Train {
        #[arg(short, long, default_value = "data/moby_dick.txt")]
        data_files: Vec<String>,
        #[arg(short, long, default_value = "16")]
        batch_size: usize,
        #[arg(long, default_value = "100")]
        num_epochs: usize,
        #[arg(short, long, default_value = "checkpoints")]
        out_dir: String,
        #[arg(long, default_value = "100")]
        save_every_n_steps: usize,

        #[arg(long, default_value = "384")]
        embed_dim: usize,
        #[arg(long, default_value = "4")]
        num_heads: usize,
        #[arg(long, default_value = "4")]
        num_layers: usize,
        #[arg(long, default_value = "768")]
        ff_hidden_dim: usize,
        #[arg(long, default_value = "128")]
        seq_length: usize,
        #[arg(long, default_value = "true")]
        qkv_bias: bool,
        #[arg(long, default_value = "0.01")]
        dropout: f32,
        #[arg(long, default_value = "5e-4")]
        learning_rate: f32,
        #[arg(long, default_value = "5e-3")]
        weight_decay: f32,
    },
    Chat {
        #[arg(short, long, default_value = "checkpoints/test.ron")]
        checkpoint: String,
        #[arg(short, long, default_value = "0.8")]
        temp: f32,
        #[arg(short, long, default_value = "1")]
        top_k: usize,
    },
}

fn main() {
    let args = CLIArgs::parse();

    match args.training {
        SubCommands::Train {
            data_files,
            batch_size,
            out_dir,
            num_epochs,
            save_every_n_steps,
            ff_hidden_dim,
            embed_dim,
            num_heads,
            num_layers,
            seq_length,
            qkv_bias,
            dropout,
            learning_rate,
            weight_decay,
        } => {
            println!("Using data file: {}", data_files.join(", "));

            let mut data_loader = data::DataLoader::new(data_files, seq_length, batch_size)
                .expect("Failed to load data");

            let tokenizer = ByteTokenizer;

            let config = ModelConfig {
                vocab_size: tokenizer.vocab_size(),
                ff_hidden_dim,
                embed_dim,
                attention_heads: num_heads,
                transformer_blocks: num_layers,
                seq_length,
                qkv_bias,
                dropout,
                learning_rate,
                weight_decay,
            };

            let mut model = model::CrabformerModel::new(config).expect("Failed to create model");

            model
                .train(&mut data_loader, num_epochs, save_every_n_steps, &out_dir)
                .expect("Failed to train model");

            // Save final model weights
            let file_name = format!("{}/model.ron", out_dir);

            if let Err(e) = model.save_weights(&out_dir, &file_name) {
                eprintln!("Failed to save model weights: {}", e);
            } else {
                println!(
                    "=====\nTraining complete!\nFinal model saved to '{}'",
                    file_name
                );
            }
        }
        SubCommands::Chat {
            checkpoint,
            temp,
            top_k,
        } => {
            println!("Using checkpoint: {}", checkpoint);
            let model = model::CrabformerModel::load(checkpoint).expect("Failed to load model");
            generate::chat(&model, temp, top_k);
        }
    }
}
