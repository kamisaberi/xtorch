#include "include/dropouts/rnn_drop.h"



#include <torch/torch.h>
#include <ostream> // For std::ostream
#include <vector>  // For std::vector

struct RNNDropImpl : torch::nn::Module {
    double p_drop_; // Probability of an element being zeroed out.
    double epsilon_ = 1e-7; // For numerical stability in division

    RNNDropImpl(double p_drop = 0.5) : p_drop_(p_drop) {
        TORCH_CHECK(p_drop_ >= 0.0 && p_drop_ <= 1.0, "Dropout probability p_drop must be between 0 and 1.");
    }

    // Input x can be:
    // - (Batch, Features): For input to an RNN cell, or output from an RNN cell, or initial hidden state.
    // - (Batch, SeqLen, Features): For an entire sequence input/output, where dropout should be consistent
    //                              across SeqLen for each batch item.
    // - (Features): If batch_size=1 and squeezed.
    torch::Tensor forward(const torch::Tensor& x) {
        if (!this->is_training() || p_drop_ == 0.0) {
            return x; // No dropout if not training or p_drop is 0
        }
        if (p_drop_ == 1.0) {
            return torch::zeros_like(x); // Drop everything if p_drop is 1
        }

        double keep_prob = 1.0 - p_drop_;
        torch::Tensor mask;

        if (x.dim() == 3) { // Input shape: (Batch, SeqLen, Features)
            // Create a mask of shape (Batch, 1, Features) to be broadcast across SeqLen.
            // This ensures the same features are dropped for all time steps of a given sample.
            std::vector<int64_t> mask_shape = {x.size(0), 1, x.size(2)};
            mask = torch::bernoulli(
                torch::full(mask_shape, keep_prob, x.options())
            ).to(x.dtype());
        } else if (x.dim() == 2) { // Input shape: (Batch, Features) or (SeqLen, Features)
            // Apply standard dropout mask.
            mask = torch::bernoulli(
                torch::full_like(x, keep_prob)
            ).to(x.dtype());
        } else if (x.dim() == 1) { // Input shape: (Features)
            // Apply standard dropout mask.
            mask = torch::bernoulli(
                torch::full_like(x, keep_prob)
            ).to(x.dtype());
        }
        else {
            TORCH_CHECK(false, "RNNDrop expects input of dim 1, 2 or 3. Got dim: ", x.dim());
            return x; // Should not reach here due to TORCH_CHECK
        }

        // Apply mask and scale (inverted dropout)
        return (x * mask) / (keep_prob + epsilon_);
    }

    void pretty_print(std::ostream& stream) const override {
        stream << "RNNDrop(p_drop=" << p_drop_ << ")";
    }
};

TORCH_MODULE(RNNDrop); // Creates the RNNDrop module "class"

/*
// Example of how to use the RNNDrop module:
// (This is for illustration and would typically be in your main application code)

#include <iostream>

// A conceptual RNN model using RNNDrop
struct MyRNNModel : torch::nn::Module {
    torch::nn::Linear input_layer;
    torch::nn::Linear recurrent_layer;
    torch::nn::Linear output_layer;
    RNNDrop rnn_drop_input;
    RNNDrop rnn_drop_hidden; // For recurrent connections

    MyRNNModel(int input_size, int hidden_size, int output_size, double p_drop_val = 0.2)
        : input_layer(torch::nn::LinearOptions(input_size, hidden_size)),
          recurrent_layer(torch::nn::LinearOptions(hidden_size, hidden_size)),
          output_layer(torch::nn::LinearOptions(hidden_size, output_size)),
          rnn_drop_input(p_drop_val), // Dropout for input transformations
          rnn_drop_hidden(p_drop_val)  // Dropout for hidden state (applied before recurrent op)
    {
        register_module("input_layer", input_layer);
        register_module("recurrent_layer", recurrent_layer);
        register_module("output_layer", output_layer);
        register_module("rnn_drop_input", rnn_drop_input);
        register_module("rnn_drop_hidden", rnn_drop_hidden);
    }

    // Process a sequence of inputs
    // inputs shape: (Batch, SeqLen, InputSize)
    // h_initial shape: (Batch, HiddenSize)
    torch::Tensor forward(const torch::Tensor& inputs, torch::Tensor h_prev) {
        std::vector<torch::Tensor> outputs_over_time;
        int64_t seq_len = inputs.size(1);

        // --- Potentially apply RNNDrop to the entire input sequence at once ---
        // This would drop the same input features across all time steps for a given batch item.
        // torch::Tensor processed_inputs = rnn_drop_input(inputs); // inputs (B, S, I) -> (B, S, I)

        for (int64_t t = 0; t < seq_len; ++t) {
            torch::Tensor x_t = inputs.select(1, t); // Get current time step input (B, I)

            // Apply dropout to the input at current time step (mask changes per time step)
            torch::Tensor x_t_dropped = rnn_drop_input(x_t); // x_t_dropped (B, I)

            torch::Tensor h_input_transformed = input_layer(x_t_dropped); // (B, H)

            // Apply dropout to the previous hidden state (mask changes per time step if h_prev is from outside)
            // For true recurrent dropout (same mask over time on h_prev), the rnn_drop_hidden
            // would need to be stateful or the mask managed externally.
            // If rnn_drop_hidden gets h_prev (B,H), it creates a (B,H) mask.
            // If we want same mask over time for h_prev, we should generate one (B,H) mask
            // and apply it to h_prev inside this loop.
            // The current RNNDropImpl, if given h_prev of (B,H), will make a (B,H) mask.
            // Let's assume for this example, we apply RNNDrop to the *transformed* hidden state
            // to demonstrate its usage.
            torch::Tensor h_prev_transformed = recurrent_layer(h_prev); // (B, H)
            torch::Tensor h_prev_transformed_dropped = rnn_drop_hidden(h_prev_transformed); // (B,H)

            h_prev = torch::tanh(h_input_transformed + h_prev_transformed_dropped); // (B,H)
            outputs_over_time.push_back(output_layer(h_prev));
        }
        return torch::stack(outputs_over_time, 1); // (B, S, OutputSize)
    }
};
TORCH_MODULE(MyRNNModel);


void run_rnn_drop_example() {
    torch::manual_seed(1); // For reproducible results

    RNNDrop dropout_module(0.5);
    dropout_module->train(); // Enable dropout

    // Test case 1: Input (Batch, Features) - like a single time step's activations
    torch::Tensor input_2d = torch::ones({2, 4}); // Batch=2, Features=4
    std::cout << "--- Test with 2D input (Batch, Features) ---" << std::endl;
    std::cout << "Input 2D:\n" << input_2d << std::endl;
    torch::Tensor output_2d_pass1 = dropout_module->forward(input_2d);
    std::cout << "Output 2D (pass 1):\n" << output_2d_pass1 << std::endl; // Mask 1
    torch::Tensor output_2d_pass2 = dropout_module->forward(input_2d);
    std::cout << "Output 2D (pass 2):\n" << output_2d_pass2 << std::endl; // Mask 2 (likely different)
    // Behavior: Standard dropout, mask changes per call.

    // Test case 2: Input (Batch, SeqLen, Features) - like a full sequence
    torch::Tensor input_3d = torch::ones({2, 3, 4}); // Batch=2, SeqLen=3, Features=4
    std::cout << "\n--- Test with 3D input (Batch, SeqLen, Features) ---" << std::endl;
    std::cout << "Input 3D (summed over features for brevity, per (Batch, SeqLen)):\n"
              << input_3d.sum(-1) << std::endl;
    torch::Tensor output_3d = dropout_module->forward(input_3d);
    std::cout << "Output 3D (summed over features):\n" << output_3d.sum(-1) << std::endl;

    // Verify consistency across SeqLen for the first batch item
    torch::Tensor first_batch_output_seq = output_3d[0]; // Shape (SeqLen, Features)
    bool is_consistent = true;
    if (first_batch_output_seq.size(0) > 1) { // If SeqLen > 1
        // Check if the pattern of zeros (dropped features) is the same across time steps
        torch::Tensor first_step_mask_pattern = (first_batch_output_seq[0] != 0);
        for (int64_t s = 1; s < first_batch_output_seq.size(0); ++s) {
            if (!torch::allclose(first_step_mask_pattern, (first_batch_output_seq[s] != 0))) {
                is_consistent = false;
                break;
            }
        }
    }
    std::cout << "Dropout mask for first batch item consistent across SeqLen: "
              << (is_consistent ? "Yes" : "No") << std::endl;
    // Expected: Yes, because a (Batch, 1, Features) mask was broadcast.

    // --- Evaluation mode ---
    dropout_module->eval();
    torch::Tensor output_3d_eval = dropout_module->forward(input_3d);
    std::cout << "\nOutput 3D (evaluation mode, summed over features):\n" << output_3d_eval.sum(-1) << std::endl;
    TORCH_CHECK(torch::allclose(input_3d, output_3d_eval), "RNNDrop eval output mismatch!");


    // --- Test with MyRNNModel (conceptual) ---
    // This example is more illustrative of where RNNDrop instances might be placed.
    // The MyRNNModel's forward pass does not strictly implement the "same mask over time for h_prev"
    // unless rnn_drop_hidden was stateful or managed its mask externally based on sequence progression.
    // The current RNNDropImpl, when given h_prev of (B,H) to rnn_drop_hidden, would generate a new (B,H) mask each time.
    // A truly variational RNN dropout on h_prev would require generating one (B,H) mask per sequence
    // and reusing it for h_prev in the loop.
    std::cout << "\n--- MyRNNModel conceptual test ---" << std::endl;
    MyRNNModel rnn_model(5, 10, 2, 0.3);
    rnn_model->train();
    torch::Tensor model_inputs = torch::randn({2, 7, 5}); // B, S, I
    torch::Tensor model_h_init = torch::zeros({2, 10});   // B, H
    torch::Tensor model_output = rnn_model->forward(model_inputs, model_h_init);
    std::cout << "MyRNNModel output shape: " << model_output.sizes() << std::endl;
}

// int main() {
//    run_rnn_drop_example();
//    return 0;
// }
*/


namespace xt::dropouts
{
    torch::Tensor rnn_drop(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto RnnDrop::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::rnn_drop(torch::zeros(10));
    }
}
