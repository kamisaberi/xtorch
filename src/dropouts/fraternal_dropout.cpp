#include "include/dropouts/fraternal_dropout.h"



// #include <torch/torch.h>
// #include <tuple>   // For std::tuple
// #include <ostream> // For std::ostream
//
// struct FraternalDropoutImpl : torch::nn::Module {
//     double p_drop_; // Probability of an element being zeroed in the primary mask
//     double p_same_mask_; // Probability that the beta network uses the same mask as alpha
//                          // (1 - p_same_mask_) is the probability beta uses the inverse mask.
//     double epsilon_ = 1e-7;
//
//     FraternalDropoutImpl(double p_drop = 0.5, double p_same_mask = 0.5)
//         : p_drop_(p_drop), p_same_mask_(p_same_mask) {
//         TORCH_CHECK(p_drop_ >= 0.0 && p_drop_ <= 1.0, "p_drop must be between 0 and 1.");
//         TORCH_CHECK(p_same_mask_ >= 0.0 && p_same_mask_ <= 1.0, "p_same_mask must be between 0 and 1.");
//     }
//
//     // This forward method takes two input tensors, one for each fraternal network branch.
//     std::tuple<torch::Tensor, torch::Tensor> forward(
//         const torch::Tensor& input_alpha,
//         const torch::Tensor& input_beta) {
//
//         TORCH_CHECK(input_alpha.sizes() == input_beta.sizes(),
//                     "Input tensors for FraternalDropout must have the same shape.");
//
//         if (!this->is_training() || p_drop_ == 0.0) {
//             return {input_alpha, input_beta}; // No dropout if not training or p_drop is 0
//         }
//
//         double keep_prob = 1.0 - p_drop_;
//
//         // 1. Create the primary dropout mask (mask_alpha)
//         // This mask determines which elements are kept (1) or dropped (0) for the alpha network.
//         torch::Tensor mask_alpha_binary = torch::bernoulli(
//             torch::full_like(input_alpha, keep_prob)
//         ).to(input_alpha.dtype());
//
//         // 2. Decide if beta network uses the same mask or the inverse mask
//         torch::Tensor mask_beta_binary;
//         if (torch::rand({1}, input_alpha.options()).item<double>() < p_same_mask_) {
//             // Beta uses the same mask as alpha
//             mask_beta_binary = mask_alpha_binary;
//         } else {
//             // Beta uses the inverse mask of alpha
//             // Inverse means what alpha drops, beta keeps, and vice-versa.
//             mask_beta_binary = 1.0 - mask_alpha_binary;
//         }
//
//         torch::Tensor output_alpha, output_beta;
//
//         // Apply dropout to alpha network (inverted dropout)
//         if (p_drop_ == 1.0) { // If alpha drops everything
//             output_alpha = torch::zeros_like(input_alpha);
//         } else {
//             output_alpha = (input_alpha * mask_alpha_binary) / (keep_prob + epsilon_);
//         }
//
//         // Apply dropout to beta network
//         // If beta uses the inverse mask, its effective "keep probability" for scaling
//         // is (1 - keep_prob) = p_drop_.
//         // If beta uses the same mask, its effective keep probability is 'keep_prob'.
//
//         if (mask_beta_binary.is_same(mask_alpha_binary)) { // Beta uses same mask
//             if (p_drop_ == 1.0) { // If alpha (and thus beta) drops everything
//                  output_beta = torch::zeros_like(input_beta);
//             } else {
//                  output_beta = (input_beta * mask_beta_binary) / (keep_prob + epsilon_);
//             }
//         } else { // Beta uses inverse mask
//             if (keep_prob == 1.0) { // If alpha keeps everything (p_drop_ == 0, handled at start)
//                                     // then inverse mask drops everything for beta.
//                  output_beta = torch::zeros_like(input_beta);
//             } else if (keep_prob == 0.0) { // If alpha drops everything (p_drop_ == 1.0)
//                                            // then inverse mask keeps everything for beta.
//                  output_beta = input_beta / (p_drop_ + epsilon_); // Scaled by 1/1.0
//             }
//              else {
//                  // Effective keep probability for beta is p_drop (since it keeps what alpha drops)
//                  output_beta = (input_beta * mask_beta_binary) / (p_drop_ + epsilon_);
//             }
//         }
//
//         return {output_alpha, output_beta};
//     }
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "FraternalDropout(p_drop=" << p_drop_
//                << ", p_same_mask=" << p_same_mask_ << ")";
//     }
// };
//
// TORCH_MODULE(FraternalDropout);
//
// /*
// // --- Example: How FraternalDropout might be used within a twin network structure ---
// // This is for context and not part of the FraternalDropoutImpl module itself.
//
// struct TwinNetworkBranch : torch::nn::Module {
//     torch::nn::Linear fc1{nullptr}, fc2{nullptr};
//     // No dropout module here, it will be applied externally or by a parent module
//
//     TwinNetworkBranch(int64_t in_features, int64_t hidden_features, int64_t out_features) {
//         fc1 = register_module("fc1", torch::nn::Linear(in_features, hidden_features));
//         fc2 = register_module("fc2", torch::nn::Linear(hidden_features, out_features));
//     }
//
//     torch::Tensor forward(const torch::Tensor& x) {
//         auto out = torch::relu(fc1(x));
//         // Placeholder for where FraternalDropout would be applied to 'out'
//         // before passing to fc2, or to the output of fc2.
//         out = fc2(out);
//         return out;
//     }
// };
// TORCH_MODULE(TwinNetworkBranch);
//
//
// struct FraternalSiameseModel : torch::nn::Module {
//     TwinNetworkBranch branch_alpha{nullptr};
//     TwinNetworkBranch branch_beta{nullptr};
//     FraternalDropout fraternal_dropout_module; // Instance of FraternalDropout
//
//     FraternalSiameseModel(int64_t in_feat, int64_t hid_feat, int64_t out_feat,
//                           double p_drop_val = 0.5, double p_same_mask_val = 0.5)
//         : fraternal_dropout_module(p_drop_val, p_same_mask_val) {
//         branch_alpha = register_module("branch_alpha", TwinNetworkBranch(in_feat, hid_feat, out_feat));
//         branch_beta = register_module("branch_beta", TwinNetworkBranch(in_feat, hid_feat, out_feat));
//     }
//
//     // Takes two inputs, one for each branch
//     std::tuple<torch::Tensor, torch::Tensor> forward(
//         const torch::Tensor& input1,
//         const torch::Tensor& input2) {
//
//         torch::Tensor x1 = torch::relu(branch_alpha->fc1(input1));
//         torch::Tensor x2 = torch::relu(branch_beta->fc1(input2));
//
//         // Apply Fraternal Dropout after the first linear layer of each branch
//         torch::Tensor x1_dropped, x2_dropped;
//         if (this->is_training()) { // FraternalDropout itself also checks this
//             std::tie(x1_dropped, x2_dropped) = fraternal_dropout_module(x1, x2);
//         } else {
//             x1_dropped = x1;
//             x2_dropped = x2;
//         }
//
//         torch::Tensor out1 = branch_alpha->fc2(x1_dropped);
//         torch::Tensor out2 = branch_beta->fc2(x2_dropped);
//
//         return {out1, out2};
//     }
// };
// TORCH_MODULE(FraternalSiameseModel);
//
//
// #include <iostream>
// void run_fraternal_dropout_example() {
//     torch::manual_seed(0);
//
//     // Test FraternalDropout module directly
//     FraternalDropout fd_op(0.5, 0.5); // 50% drop, 50% chance beta uses same mask
//     fd_op->train();
//
//     torch::Tensor in_a = torch::ones({2, 3});
//     torch::Tensor in_b = torch::ones({2, 3}) * 2.0; // Different values for beta input
//
//     std::cout << "Input Alpha (ones):\n" << in_a << std::endl;
//     std::cout << "Input Beta (twos):\n" << in_b << std::endl;
//
//     for (int i=0; i < 5; ++i) { // Run a few times to see different mask combinations
//         torch::Tensor out_a, out_b;
//         std::cout << "\n--- Iteration " << i << " ---" << std::endl;
//         std::tie(out_a, out_b) = fd_op->forward(in_a, in_b);
//         std::cout << "Output Alpha:\n" << out_a << std::endl;
//         std::cout << "Output Beta:\n" << out_b << std::endl;
//
//         // Check correlation (qualitatively)
//         // If mask_alpha has a zero, out_a will have a zero at that position.
//         // If beta used same mask, out_b will also have a zero.
//         // If beta used inverse mask, out_b will have a non-zero (scaled) there.
//     }
//
//     fd_op->eval();
//     torch::Tensor out_a_eval, out_b_eval;
//     std::tie(out_a_eval, out_b_eval) = fd_op->forward(in_a, in_b);
//     std::cout << "\n--- Evaluation Mode ---" << std::endl;
//     std::cout << "Output Alpha (eval):\n" << out_a_eval << std::endl;
//     std::cout << "Output Beta (eval):\n" << out_b_eval << std::endl;
//     TORCH_CHECK(torch::allclose(in_a, out_a_eval), "Fraternal eval alpha mismatch");
//     TORCH_CHECK(torch::allclose(in_b, out_b_eval), "Fraternal eval beta mismatch");
//
//
//     // Test with the FraternalSiameseModel
//     std::cout << "\n--- Fraternal Siamese Model Test ---" << std::endl;
//     FraternalSiameseModel siamese_model(3, 5, 2, 0.3, 0.7); // p_drop=0.3, p_same_mask=0.7
//     siamese_model->train();
//
//     torch::Tensor s_in1 = torch::randn({4,3});
//     torch::Tensor s_in2 = torch::randn({4,3});
//     torch::Tensor s_out1, s_out2;
//     std::tie(s_out1, s_out2) = siamese_model->forward(s_in1, s_in2);
//     std::cout << "Siamese Model Output 1 (train) shape: " << s_out1.sizes() << std::endl;
//     std::cout << "Siamese Model Output 2 (train) shape: " << s_out2.sizes() << std::endl;
//
//     siamese_model->eval();
//     std::tie(s_out1, s_out2) = siamese_model->forward(s_in1, s_in2);
//     std::cout << "Siamese Model Output 1 (eval) shape: " << s_out1.sizes() << std::endl;
//
// }
//
// // int main() {
// //    run_fraternal_dropout_example();
// //    return 0;
// // }
// */
//


namespace xt::dropouts
{
    torch::Tensor fraternal_dropout(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto FraternalDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::fraternal_dropout(torch::zeros(10));
    }
}
