mod adamw;
mod dashboard;
mod data;
mod errors;
mod generate;
mod layers;
mod loss;
mod metrics;
mod model;
mod params;
mod tokenizer;

use clap::{Parser, Subcommand};

use crate::{
    params::{BATCH_SIZE, NUM_EPOCHS},
    tokenizer::ByteTokenizer,
};

#[derive(Parser)]
struct CLIArgs {
    // #[arg(short, long, default_value = "data/moby_dick.txt")]
    #[clap(subcommand)]
    training: SubCommands,
}

#[derive(Subcommand)]
enum SubCommands {
    Train {
        #[arg(short, long, default_value = "data/moby_dick.txt")]
        data_files: Vec<String>,
    },
    Chat {
        #[arg(short, long, default_value = "checkpoints/test.ron")]
        checkpoint: String,
    },
}

fn main() {
    let args = CLIArgs::parse();

    match args.training {
        SubCommands::Train { data_files } => {
            println!("Using data file: {}", data_files.join(", "));
            let mut data_loader =
                data::DataLoader::new(data_files, BATCH_SIZE).expect("Failed to load data");

            let tokenizer = ByteTokenizer;
            let vocab_size = tokenizer.vocab_size();

            let mut model =
                model::CrabformerModel::new(vocab_size).expect("Failed to create model");
            model
                .train(&mut data_loader, NUM_EPOCHS)
                .expect("Failed to train model");
        }
        SubCommands::Chat { checkpoint } => {
            println!("Using checkpoint: {}", checkpoint);
            let model = model::CrabformerModel::load(checkpoint).expect("Failed to load model");
            generate::chat(&model);
        }
    }
}
