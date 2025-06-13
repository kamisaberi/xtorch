#include "include/dropouts/zone_out.h"


// #include <torch/torch.h>
// #include <ostream> // For std::ostream
//
// struct ZoneoutImpl : torch::nn::Module {
//     double p_zoneout_h_; // Probability of a hidden unit's state being "zoned out" (copied from previous step)
//                          // This is often denoted as 'zh' in the paper for hidden state zoneout.
//                          // There can also be 'zc' for cell state zoneout in LSTMs. This module handles one state tensor.
//
//     ZoneoutImpl(double p_zoneout_h = 0.1) : p_zoneout_h_(p_zoneout_h) {
//         TORCH_CHECK(p_zoneout_h_ >= 0.0 && p_zoneout_h_ <= 1.0,
//                     "Zoneout probability p_zoneout_h must be between 0 and 1.");
//     }
//
//     // prev_state_h: The hidden state from the previous time step (t-1).
//     // current_state_h_candidate: The newly computed hidden state for time t, before Zoneout.
//     // Both tensors are expected to have the same shape, e.g., (Batch, HiddenSize).
//     torch::Tensor forward(const torch::Tensor& prev_state_h, const torch::Tensor& current_state_h_candidate) {
//         TORCH_CHECK(prev_state_h.sizes() == current_state_h_candidate.sizes(),
//                     "prev_state_h and current_state_h_candidate must have the same shape for Zoneout.");
//
//         if (!this->is_training() || p_zoneout_h_ == 0.0) {
//             // If not training or no zoneout, use the newly computed candidate state.
//             return current_state_h_candidate;
//         }
//         if (p_zoneout_h_ == 1.0) {
//             // If p_zoneout is 1.0, always copy the previous state.
//             return prev_state_h;
//         }
//
//         // Generate a binary mask.
//         // Where mask is 1, we "zone out" (copy prev_state_h).
//         // Where mask is 0, we update (use current_state_h_candidate).
//         torch::Tensor zoneout_mask = torch::bernoulli(
//             torch::full_like(prev_state_h, p_zoneout_h_) // Probability of zoning out (keeping previous)
//         ).to(prev_state_h.dtype()); // Ensure mask has same dtype for multiplication
//
//         // Combine states:
//         // zoned_out_state = zoneout_mask * prev_state_h + (1 - zoneout_mask) * current_state_h_candidate;
//         // The paper (Section 2.1) defines it as:
//         // h_t = d_t * h_t-1 + (1 - d_t) * h_hat_t
//         // where d_t is the binary mask (1 for zoneout, 0 for update), drawn with prob p_zoneout_h.
//         // So, our zoneout_mask corresponds to d_t.
//
//         torch::Tensor next_state_h = (zoneout_mask * prev_state_h) +
//                                      ((1.0 - zoneout_mask) * current_state_h_candidate);
//
//         return next_state_h;
//     }
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "Zoneout(p_zoneout_h=" << p_zoneout_h_ << ")";
//     }
// };
//
// TORCH_MODULE(Zoneout);
//
// /*
// // --- Example: How Zoneout might be used in a custom RNN cell ---
// // This is for context and not part of the ZoneoutImpl module itself.
//
// struct CustomRNNCellWithZoneoutImpl : torch::nn::Module {
//     torch::nn::Linear W_ih{nullptr}, W_hh{nullptr};
//     Zoneout zoneout_h_module; // For hidden state h
//
//     // For LSTMs, you'd have another Zoneout module for the cell state 'c'.
//     // Zoneout zoneout_c_module;
//
//     CustomRNNCellWithZoneoutImpl(int input_size, int hidden_size, double p_zoneout_val = 0.1)
//         : zoneout_h_module(p_zoneout_val) {
//         W_ih = register_module("W_ih", torch::nn::Linear(input_size, hidden_size));
//         W_hh = register_module("W_hh", torch::nn::Linear(hidden_size, hidden_size));
//         // zoneout_h_module is already initialized
//     }
//
//     // h_prev_actual shape: (Batch, HiddenSize) - actual state from t-1
//     // x_t    shape: (Batch, InputSize) - input at current time t
//     std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& x_t, const torch::Tensor& h_prev_actual) {
//         // Calculate the candidate for the new hidden state h_hat_t
//         torch::Tensor h_hat_t_candidate = torch::tanh(W_ih(x_t) + W_hh(h_prev_actual));
//
//         // Apply Zoneout to determine the actual h_t
//         // The Zoneout module itself checks this->is_training()
//         torch::Tensor h_next_actual = zoneout_h_module(h_prev_actual, h_hat_t_candidate);
//
//         // In a simple RNN, output is often the hidden state.
//         // In LSTM/GRU, output might differ or be processed further.
//         return {h_next_actual, h_next_actual}; // Return (output_t, h_t_next_actual)
//     }
// };
// TORCH_MODULE(CustomRNNCellWithZoneout);
//
//
// #include <iostream>
// #include <iomanip> // For std::fixed, std::setprecision
//
// void run_zoneout_example() {
//     torch::manual_seed(0);
//     std::cout << std::fixed << std::setprecision(4);
//
//     double prob_zoneout = 0.5; // 50% chance of a unit keeping its previous state
//
//     // Test Zoneout module directly
//     Zoneout zo_direct(prob_zoneout);
//     std::cout << "Zoneout Module (direct use): " << zo_direct << std::endl;
//
//     torch::Tensor h_t_minus_1 = torch::tensor({{1.0, 2.0, 3.0}, {10.0, 20.0, 30.0}}, torch::kFloat32); // Batch=2, Hidden=3
//     torch::Tensor h_t_candidate = torch::tensor({{0.1, 0.2, 0.3}, {-1.0, -2.0, -3.0}}, torch::kFloat32);
//
//     std::cout << "\nh_t_minus_1:\n" << h_t_minus_1 << std::endl;
//     std::cout << "h_t_candidate (newly computed state for t):\n" << h_t_candidate << std::endl;
//
//     // --- Training mode ---
//     zo_direct->train();
//     std::cout << "\n--- Training Mode (Direct Use) ---" << std::endl;
//     for (int i = 0; i < 3; ++i) { // Run a few times to see different zoneout patterns
//         torch::Tensor h_t_actual = zo_direct(h_t_minus_1, h_t_candidate);
//         std::cout << "Run " << i << ": h_t_actual (after zoneout):\n" << h_t_actual << std::endl;
//         // Expected: Some elements in h_t_actual will be from h_t_minus_1, others from h_t_candidate.
//     }
//
//     // --- Evaluation mode ---
//     zo_direct->eval();
//     torch::Tensor h_t_eval = zo_direct(h_t_minus_1, h_t_candidate);
//     std::cout << "\n--- Evaluation Mode (Direct Use) ---" << std::endl;
//     std::cout << "h_t_actual (eval mode):\n" << h_t_eval << std::endl;
//     // Expected: h_t_eval should be identical to h_t_candidate (no zoneout).
//     TORCH_CHECK(torch::allclose(h_t_eval, h_t_candidate), "Zoneout eval output mismatch!");
//
//
//     // --- Test with CustomRNNCellWithZoneout ---
//     std::cout << "\n--- CustomRNNCellWithZoneout Test ---" << std::endl;
//     int input_dim = 4, hidden_dim = 3, batch_s = 2;
//     CustomRNNCellWithZoneout rnn_cell(input_dim, hidden_dim, prob_zoneout);
//     std::cout << "Custom RNN Cell Module: " << rnn_cell << std::endl;
//
//     torch::Tensor x_input_step = torch::randn({batch_s, input_dim});
//     torch::Tensor h_prev_for_cell = torch::randn({batch_s, hidden_dim}); // Initial hidden state
//
//     rnn_cell->train();
//     auto [cell_output_train, cell_h_next_train] = rnn_cell(x_input_step, h_prev_for_cell);
//     std::cout << "RNN Cell h_prev_for_cell:\n" << h_prev_for_cell << std::endl;
//     std::cout << "RNN Cell h_next_actual (train):\n" << cell_h_next_train << std::endl;
//     // Some elements in cell_h_next_train should match h_prev_for_cell due to zoneout.
//
//     rnn_cell->eval();
//     auto [cell_output_eval, cell_h_next_eval] = rnn_cell(x_input_step, h_prev_for_cell);
//     std::cout << "RNN Cell h_next_actual (eval):\n" << cell_h_next_eval << std::endl;
//     // In eval, cell_h_next_eval should be the result of tanh(W_ih*x + W_hh*h_prev) without zoneout.
// }
//
// // int main() {
// //    run_zoneout_example();
// //    return 0;
// // }
// */


namespace xt::dropouts
{
    ZoneOut::ZoneOut(double p_zoneout_h) : p_zoneout_h_(p_zoneout_h)
    {
        TORCH_CHECK(p_zoneout_h_ >= 0.0 && p_zoneout_h_ <= 1.0,
                    "Zoneout probability p_zoneout_h must be between 0 and 1.");
    }


    auto ZoneOut::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        vector<std::any> tensors_ = tensors;
        auto prev_state_h = std::any_cast<torch::Tensor>(tensors_[0]);
        auto current_state_h_candidate = std::any_cast<torch::Tensor>(tensors_[1]);


        TORCH_CHECK(prev_state_h.sizes() == current_state_h_candidate.sizes(),
                    "prev_state_h and current_state_h_candidate must have the same shape for Zoneout.");

        if (!this->is_training() || p_zoneout_h_ == 0.0)
        {
            // If not training or no zoneout, use the newly computed candidate state.
            return current_state_h_candidate;
        }
        if (p_zoneout_h_ == 1.0)
        {
            // If p_zoneout is 1.0, always copy the previous state.
            return prev_state_h;
        }

        // Generate a binary mask.
        // Where mask is 1, we "zone out" (copy prev_state_h).
        // Where mask is 0, we update (use current_state_h_candidate).
        torch::Tensor zoneout_mask = torch::bernoulli(
            torch::full_like(prev_state_h, p_zoneout_h_) // Probability of zoning out (keeping previous)
        ).to(prev_state_h.dtype()); // Ensure mask has same dtype for multiplication

        // Combine states:
        // zoned_out_state = zoneout_mask * prev_state_h + (1 - zoneout_mask) * current_state_h_candidate;
        // The paper (Section 2.1) defines it as:
        // h_t = d_t * h_t-1 + (1 - d_t) * h_hat_t
        // where d_t is the binary mask (1 for zoneout, 0 for update), drawn with prob p_zoneout_h.
        // So, our zoneout_mask corresponds to d_t.

        torch::Tensor next_state_h = (zoneout_mask * prev_state_h) +
            ((1.0 - zoneout_mask) * current_state_h_candidate);

        return next_state_h;
    }
}
