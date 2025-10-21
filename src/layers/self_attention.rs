// use ndarray::{Array2, Array3, Axis};
// use rand::{rngs::StdRng, seq};

// use crate::{
//     layers::{dropout::Dropout, normalization::Normalized, xavier_initialized_array},
//     params::get_rng,
// };

// pub struct SelfAttentionLayer {
//     pub query_weights: Array2<f32>,
//     pub key_weights: Array2<f32>,
//     pub value_weights: Array2<f32>,

//     use_casual_mask: bool,
// }

// // Attention layer block.
// /// The goal of an attention layer is to compute a context vector for each token in the input sequence by "attending" to all other tokens in the same sequence.
// ///
// /// Notes:
// /// To get the attention weights we compute the dot product of the query with all keys, divide each by sqrt(d_k) (where d_k is the dimension of the key vectors), and apply a softmax function to obtain the weights on the values.
// /// The output of the attention layer is a weighted sum of the value vectors, where the weights are given by the attention weights computed from the queries and keys.
// ///
// /// Remebmer to also use the square of the dimension of the key vectors when scaling the dot products to avoid large values that can lead to vanishing gradients during training.

// impl SelfAttentionLayer {
//     pub fn new(dim: usize, seed: Option<u64>) -> Self {
//         let mut rng = get_rng(seed);

//         let query_weights = xavier_initialized_array(dim, dim, &mut rng);
//         let key_weights = xavier_initialized_array(dim, dim, &mut rng);
//         let value_weights = xavier_initialized_array(dim, dim, &mut rng);

//         Self {
//             query_weights,
//             key_weights,
//             value_weights,
//             use_casual_mask: false,
//         }
//     }

//     pub fn with_casual_mask(self) -> Self {
//         Self {
//             use_casual_mask: true,
//             ..self
//         }
//     }

//     fn causal_mask(&self, shape: (usize, usize)) -> Array2<f32> {
//         Array2::from_shape_fn(shape, |(i, j)| if j <= i { 0.0 } else { f32::NEG_INFINITY })
//     }

//     // Placeholder for forward pass
//     pub fn forward(&self, input: &Array3<f32>, rng: &mut StdRng) -> Array3<f32> {
//         // input shape: (batch_size, seq_len, embed_dim)
//         // matrix shape: (embed_dim, embed_dim)
//         // output shape: (batch_size, seq_len, embed_dim)

//         let (batch_size, seq_len, embed_dim) = input.dim();

//         println!("amogus");
//         // Reshape input to 2D for matrix multiplication
//         let input_2d = input.to_shape((batch_size * seq_len, embed_dim)).unwrap();

//         // Compute queries, keys, and values with 2D dot products and reshape back to 3D on batch dimension
//         let queries_2d = input_2d.dot(&self.query_weights);
//         let keys_2d = input_2d.dot(&self.key_weights);
//         let values_2d = input_2d.dot(&self.value_weights);

//         let queries = queries_2d
//             .to_shape((batch_size, seq_len, embed_dim))
//             .unwrap();
//         let keys = keys_2d.to_shape((batch_size, seq_len, embed_dim)).unwrap();
//         let values = values_2d
//             .to_shape((batch_size, seq_len, embed_dim))
//             .unwrap();

//         let mut output = Array3::<f32>::zeros((batch_size, seq_len, embed_dim));

//         let batch_axis = Axis(0);
//         for batch in 0..batch_size {
//             // Get the queries, keys, and values for the current batch
//             let batch_queries = queries.index_axis(batch_axis, batch);
//             let batch_keys = keys.index_axis(batch_axis, batch);
//             let batch_values = values.index_axis(batch_axis, batch);

//             let attention_scores = batch_queries.dot(&batch_keys.t());

//             let attention_scores = if self.use_casual_mask {
//                 attention_scores + self.causal_mask((seq_len, seq_len))
//             } else {
//                 attention_scores
//             };

//             // Scale by sqrt(dk) to prevent large dot product values that can lead to vanishing gradients
//             let dk = batch_keys.ncols() as f32;
//             let mut attention_weights = attention_scores.mapv(|x| x / dk.sqrt());

//             // Normalize weights with softmax
//             attention_weights.softmax(0);

//             // Apply dropout to attention weights (training only)
//             attention_weights.with_dropout(0.5, rng);

//             let context_vectors = attention_weights.dot(&batch_values);

//             // Assign the context vectors to the corresponding batch in the output
//             output
//                 .index_axis_mut(batch_axis, batch)
//                 .assign(&context_vectors);
//         }

//         output
//     }
// }
