#include "include/dropouts/recurrent_dropout.h"


//
// #include <torch/torch.h>
// #include <ostream> // For std::ostream
//
// struct RecurrentDropoutImpl : torch::nn::Module {
//     double p_drop_; // Probability of an element being zeroed out.
//     double epsilon_ = 1e-7; // For numerical stability in division
//
//     // This member will store the mask if we want to reuse it across time steps.
//     // It's tricky to manage state like this in a simple nn::Module if it's called
//     // for different sequences without explicit reset.
//     // For this implementation, we'll generate a mask per call, suitable if
//     // the input `x` to `forward` is already shaped (Batch, Features) and this
//     // mask is then applied consistently across time by the calling RNN.
//     // Or, if `x` is (Batch, SeqLen, Features), and we want different masks per batch item
//     // but same mask across SeqLen for that item.
//     // This implementation will generate a mask for the (Batch, Features) part.
//     // It is up to the CALLER to ensure this mask is used consistently across time-steps for a given sequence.
//
//     RecurrentDropoutImpl(double p_drop = 0.5) : p_drop_(p_drop) {
//         TORCH_CHECK(p_drop_ >= 0.0 && p_drop_ <= 1.0, "Dropout probability p_drop must be between 0 and 1.");
//     }
//
//     // Input x is expected to be of shape (Batch, Features) or (Features) if batch_size=1
//     // or (Batch, SeqLen, Features) if applying dropout to an entire sequence with potentially
//     // different masks per batch item but same mask across sequence length.
//     // For true recurrent dropout (same mask over time for hidden-to-hidden):
//     // The mask should be generated based on (Batch, HiddenFeatures) once per sequence.
//     torch::Tensor forward(const torch::Tensor& x) {
//         if (!this->is_training() || p_drop_ == 0.0) {
//             return x;
//         }
//         if (p_drop_ == 1.0) {
//             return torch::zeros_like(x);
//         }
//
//         double keep_prob = 1.0 - p_drop_;
//
//         // Generate a dropout mask. For recurrent dropout, this mask should be
//         // consistent across the time steps for a given sequence sample.
//         // If x is (Batch, Features), the mask is (Batch, Features).
//         // If x is (Batch, SeqLen, Features), to have the same mask across SeqLen for each batch item,
//         // the mask should be generated with shape (Batch, 1, Features) and then broadcast.
//         torch::Tensor mask;
//         if (x.dim() == 3) { // Assuming (Batch, SeqLen, Features)
//             // Create mask of shape (Batch, 1, Features)
//             torch::Tensor mask_template = torch::full({x.size(0), 1, x.size(2)}, keep_prob, x.options());
//             mask = torch::bernoulli(mask_template).to(x.dtype());
//             // This mask will broadcast along the SeqLen dimension.
//         } else if (x.dim() == 2) { // Assuming (Batch, Features) or (SeqLen, Features)
//             // Create mask of shape (Batch, Features)
//             mask = torch::bernoulli(torch::full_like(x, keep_prob)).to(x.dtype());
//         } else if (x.dim() == 1) { // Assuming (Features)
//             mask = torch::bernoulli(torch::full_like(x, keep_prob)).to(x.dtype());
//         }
//         else {
//             TORCH_CHECK(false, "RecurrentDropout expects input of dim 1, 2 or 3.");
//             return x; // Should not reach here
//         }
//
//         return (x * mask) / (keep_prob + epsilon_);
//     }
//
//     // To implement true stateful recurrent dropout where the mask is generated once
//     // per sequence and reused, the RNN cell/layer would typically manage this.
//     // For example, it could have a method like:
//     // void reset_mask(c10::IntArrayRef mask_shape, torch::Device device); // Generates and stores a mask
//     // torch::Tensor apply_stored_mask(const torch::Tensor& x); // Uses the stored mask
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "RecurrentDropout(p_drop=" << p_drop_ << ")";
//     }
// };
//
// TORCH_MODULE(RecurrentDropout);
//
//
// /*
// // --- Example: How RecurrentDropout might be used in a custom RNN cell ---
// // This demonstrates the principle of reusing the mask.
//
// struct CustomRNNCellImpl : torch::nn::Module {
//     torch::nn::Linear W_ih{nullptr}, W_hh{nullptr};
//     RecurrentDropout recurrent_dropout_h; // For h_t-1 -> h_t
//     RecurrentDropout input_dropout_i;     // For x_t -> h_t (optional, can be standard dropout)
//
//     // Mask for hidden state, to be kept consistent across sequence steps
//     torch::Tensor h_dropout_mask;
//     bool needs_new_h_mask = true;
//
//
//     CustomRNNCellImpl(int input_size, int hidden_size, double p_drop_recurrent = 0.2, double p_drop_input = 0.1)
//         : recurrent_dropout_h(p_drop_recurrent), input_dropout_i(p_drop_input) {
//         W_ih = register_module("W_ih", torch::nn::Linear(input_size, hidden_size));
//         W_hh = register_module("W_hh", torch::nn::Linear(hidden_size, hidden_size));
//     }
//
//     // Call this at the beginning of each new sequence
//     void reset_sequence_state() {
//         needs_new_h_mask = true;
//     }
//
//     // h_prev shape: (Batch, HiddenSize)
//     // x_t    shape: (Batch, InputSize)
//     std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& x_t, const torch::Tensor& h_prev) {
//         // Apply dropout to input (optional, can be standard dropout too)
//         // This input_dropout can generate a new mask per time step if desired.
//         torch::Tensor x_t_dropped = input_dropout_i(x_t); // Standard dropout on input
//
//         // For recurrent connections, use the *same* dropout mask across time steps for h_prev
//         torch::Tensor h_prev_for_recurrent_dropout = h_prev;
//         if (this->is_training() && recurrent_dropout_h->p_drop_ > 0.0) {
//             if (needs_new_h_mask || !h_dropout_mask.defined()) {
//                 // Generate a new mask for h_prev (Batch, HiddenSize)
//                 // This mask will be reused for all subsequent steps of this sequence.
//                 double keep_prob = 1.0 - recurrent_dropout_h->p_drop_;
//                 if (h_prev.numel() > 0) { // Ensure h_prev is not empty
//                      h_dropout_mask = torch::bernoulli(
//                         torch::full_like(h_prev, keep_prob)
//                     ).to(h_prev.dtype());
//                 } else { // Handle empty h_prev if necessary
//                     h_dropout_mask = torch::empty_like(h_prev);
//                 }
//                 needs_new_h_mask = false;
//             }
//             // Apply the consistent mask (inverted dropout)
//             if (h_dropout_mask.numel() > 0) {
//                  h_prev_for_recurrent_dropout = (h_prev * h_dropout_mask) / ( (1.0 - recurrent_dropout_h->p_drop_) + recurrent_dropout_h->epsilon_ );
//             } else {
//                  h_prev_for_recurrent_dropout = h_prev;
//             }
//
//         } else {
//             h_prev_for_recurrent_dropout = h_prev; // No recurrent dropout if not training or p_drop is 0
//         }
//
//         torch::Tensor h_next = torch::tanh(W_ih(x_t_dropped) + W_hh(h_prev_for_recurrent_dropout));
//         return {h_next, h_next}; // In GRU/LSTM, output might differ from hidden state
//     }
// };
// TORCH_MODULE(CustomRNNCell);
//
//
// #include <iostream>
// void run_recurrent_dropout_example() {
//     torch::manual_seed(0);
//
//     // Test RecurrentDropout directly (stateless version, mask generated per call)
//     RecurrentDropout rd_direct(0.5);
//     rd_direct->train();
//     torch::Tensor r_in = torch::ones({2, 3}); // Batch=2, Features=3
//     std::cout << "Direct RecurrentDropout (stateless) on (2,3) input:" << std::endl;
//     std::cout << "Input:\n" << r_in << std::endl;
//     std::cout << "Output 1:\n" << rd_direct(r_in) << std::endl; // Mask 1
//     std::cout << "Output 2:\n" << rd_direct(r_in) << std::endl; // Mask 2 (likely different)
//
//     torch::Tensor r_in_seq = torch::ones({2, 4, 3}); // Batch=2, SeqLen=4, Features=3
//     std::cout << "\nDirect RecurrentDropout on (2,4,3) input (same mask across seq for each batch item):" << std::endl;
//     torch::Tensor r_out_seq = rd_direct(r_in_seq);
//     // Check if mask is same across sequence for first batch item
//     // Squeeze to (SeqLen, Features) for one batch item
//     torch::Tensor first_batch_item_out = r_out_seq[0];
//     bool first_item_consistent = true;
//     if (first_batch_item_out.size(0) > 1) { // if SeqLen > 1
//         for (int s=1; s < first_batch_item_out.size(0); ++s) {
//             if (!torch::allclose( (first_batch_item_out[0] != 0), (first_batch_item_out[s] != 0) )) {
//                 first_item_consistent = false;
//                 break;
//             }
//         }
//     }
//     std::cout << "Mask for first batch item consistent across seq_len: " << (first_item_consistent ? "Yes" : "No") << std::endl;
//     // std::cout << "Output seq sample 0 (sum over features):\n" << r_out_seq[0].sum(-1) << std::endl;
//
//
//     // --- Test with CustomRNNCell that manages mask state ---
//     std::cout << "\n--- CustomRNNCell with stateful Recurrent Dropout ---" << std::endl;
//     int input_dim = 5, hidden_dim = 4, batch_s = 2;
//     CustomRNNCell rnn_cell(input_dim, hidden_dim, 0.5);
//     rnn_cell->train();
//
//     torch::Tensor x_input_step1 = torch::randn({batch_s, input_dim});
//     torch::Tensor h_prev_init = torch::zeros({batch_s, hidden_dim});
//
//     // --- Sequence 1 ---
//     std::cout << "Sequence 1, Step 1:" << std::endl;
//     rnn_cell->reset_sequence_state(); // Important for new sequence
//     auto [h_next1_s1, ~] = rnn_cell->forward(x_input_step1, h_prev_init);
//     // The h_dropout_mask is now generated and stored in rnn_cell
//     // For debugging, one might try to inspect rnn_cell->h_dropout_mask here.
//     // std::cout << "RNN Cell h_dropout_mask (after step 1):\n" << rnn_cell->h_dropout_mask << std::endl;
//
//
//     torch::Tensor x_input_step2 = torch::randn({batch_s, input_dim});
//     std::cout << "Sequence 1, Step 2 (using same h_dropout_mask):" << std::endl;
//     auto [h_next1_s2, ~] = rnn_cell->forward(x_input_step2, h_next1_s1);
//     // h_next1_s1 was dropped with the *same* mask as h_prev_init would have been.
//
//     // --- Sequence 2 (demonstrate mask reset) ---
//     std::cout << "Sequence 2, Step 1 (new h_dropout_mask will be generated):" << std::endl;
//     rnn_cell->reset_sequence_state(); // New sequence, reset mask flag
//     auto [h_next2_s1, ~] = rnn_cell->forward(x_input_step1, h_prev_init);
//     // A *new* h_dropout_mask is generated for this sequence.
//     // std::cout << "RNN Cell h_dropout_mask (Sequence 2, step 1):\n" << rnn_cell->h_dropout_mask << std::endl;
//
//
//     // --- Evaluation mode (no dropout expected) ---
//     std::cout << "\n--- CustomRNNCell Evaluation Mode ---" << std::endl;
//     rnn_cell->eval();
//     rnn_cell->reset_sequence_state();
//     auto [h_next_eval, ~] = rnn_cell->forward(x_input_step1, h_prev_init);
//     // Here, h_prev should not have been dropped, regardless of any stored mask.
//     // We can check if output would be different if a non-zero mask was applied
//     // (A proper check would compare against a no-dropout version of the math).
// }
//
// // int main() {
// //    run_recurrent_dropout_example();
// //    return 0;
// // }
// */


namespace xt::dropouts
{
    RecurrentDropout::RecurrentDropout(double p_drop) : p_drop_(p_drop)
    {
        TORCH_CHECK(p_drop_ >= 0.0 && p_drop_ <= 1.0, "Dropout probability p_drop must be between 0 and 1.");
    }


    auto RecurrentDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        vector<std::any> tensors_ = tensors;
        auto x = std::any_cast<torch::Tensor>(tensors_[0]);

        if (!this->is_training() || p_drop_ == 0.0)
        {
            return x;
        }
        if (p_drop_ == 1.0)
        {
            return torch::zeros_like(x);
        }

        double keep_prob = 1.0 - p_drop_;

        // Generate a dropout mask. For recurrent dropout, this mask should be
        // consistent across the time steps for a given sequence sample.
        // If x is (Batch, Features), the mask is (Batch, Features).
        // If x is (Batch, SeqLen, Features), to have the same mask across SeqLen for each batch item,
        // the mask should be generated with shape (Batch, 1, Features) and then broadcast.
        torch::Tensor mask;
        if (x.dim() == 3)
        {
            // Assuming (Batch, SeqLen, Features)
            // Create mask of shape (Batch, 1, Features)
            torch::Tensor mask_template = torch::full({x.size(0), 1, x.size(2)}, keep_prob, x.options());
            mask = torch::bernoulli(mask_template).to(x.dtype());
            // This mask will broadcast along the SeqLen dimension.
        }
        else if (x.dim() == 2)
        {
            // Assuming (Batch, Features) or (SeqLen, Features)
            // Create mask of shape (Batch, Features)
            mask = torch::bernoulli(torch::full_like(x, keep_prob)).to(x.dtype());
        }
        else if (x.dim() == 1)
        {
            // Assuming (Features)
            mask = torch::bernoulli(torch::full_like(x, keep_prob)).to(x.dtype());
        }
        else
        {
            TORCH_CHECK(false, "RecurrentDropout expects input of dim 1, 2 or 3.");
            return x; // Should not reach here
        }

        return (x * mask) / (keep_prob + epsilon_);
    }
}
