use ndarray::{Array2, Array3};

/// Computes the cross-entropy loss between predictions and targets.
pub fn cross_entropy_loss(logits: &Array3<f32>, targets: &Array2<u32>) -> f32 {
    let (batch_size, seq_len, _vocab_size) = logits.dim();
    let mut total_loss = 0.0;

    // Iterate over each token positions in all sequences in the batch and accumulate loss
    for batch_idx in 0..batch_size {
        for seq_idx in 0..seq_len {
            // Get the logits for the current position of the sequence
            let logit_slice = logits.slice(ndarray::s![batch_idx, seq_idx, ..]);

            // Get the actual target token index
            let target = targets[[batch_idx, seq_idx]] as usize;

            // Compute softmax of logits for numerical stability.
            // We could use the Softmax trait, but it requires us to make a data copy.
            let max_logit = logit_slice
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);

            let exp_sum: f32 = logit_slice.iter().map(|&x| (x - max_logit).exp()).sum();
            let log_sum_exp = max_logit + exp_sum.ln();

            // Cross-entropy: -log(softmax(logit[target]))
            let loss = -(logit_slice[target] - log_sum_exp);
            total_loss += loss;
        }
    }

    total_loss / (batch_size * seq_len) as f32
}

/// Computes the gradient of cross-entropy loss with respect to logits.
pub fn cross_entropy_loss_backward(logits: &Array3<f32>, targets: &Array2<u32>) -> Array3<f32> {
    let (batch_size, seq_len, vocab_size) = logits.dim();
    let mut grad = Array3::<f32>::zeros((batch_size, seq_len, vocab_size));

    for batch_idx in 0..batch_size {
        for seq_idx in 0..seq_len {
            let logit_slice = logits.slice(ndarray::s![batch_idx, seq_idx, ..]);
            let target = targets[[batch_idx, seq_idx]] as usize;

            // Compute softmax
            let max_logit = logit_slice
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = logit_slice.iter().map(|&x| (x - max_logit).exp()).sum();

            // Fill gradient: softmax(logits)
            for (i, &logit) in logit_slice.iter().enumerate() {
                let softmax_i = (logit - max_logit).exp() / exp_sum;
                grad[[batch_idx, seq_idx, i]] = softmax_i;
            }

            // Subtract 1 from the target class
            grad[[batch_idx, seq_idx, target]] -= 1.0;
        }
    }

    // Average gradient over batch and sequence
    grad / (batch_size * seq_len) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, arr3};

    #[test]
    fn test_cross_entropy_loss() {
        // Simple test case: batch_size=1, seq_len=2, vocab_size=3
        let logits = arr3(&[[[1.0, 2.0, 0.5], [0.5, 1.0, 2.0]]]);
        let targets = arr2(&[[1, 2]]);

        let loss = cross_entropy_loss(&logits, &targets);
        assert!(loss > 0.0);
    }

    #[test]
    fn test_cross_entropy_gradient() {
        let logits = arr3(&[[[1.0, 2.0, 0.5], [0.5, 1.0, 2.0]]]);
        let targets = arr2(&[[1, 2]]);

        let grad = cross_entropy_loss_backward(&logits, &targets);

        // Check shape
        assert_eq!(grad.dim(), (1, 2, 3));

        // Check that gradient sums to approximately 0 for each position
        // (softmax gradient property)
        let sum_0: f32 = grad.slice(ndarray::s![0, 0, ..]).sum();
        let sum_1: f32 = grad.slice(ndarray::s![0, 1, ..]).sum();

        assert!(sum_0.abs() < 0.01);
        assert!(sum_1.abs() < 0.01);
    }
}
