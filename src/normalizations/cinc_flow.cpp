#include "include/normalizations/cinc_flow.h"

//
// #include <torch/torch.h>
// #include <iostream>
// #include <vector>
// #include <tuple> // For returning multiple values
//
// // Forward declaration for the Impl struct
// struct cIncFlowImpl;
//
// // The main module struct that users will interact with.
// struct cIncFlow : torch::nn::ModuleHolder<cIncFlowImpl> {
//     using torch::nn::ModuleHolder<cIncFlowImpl>::ModuleHolder;
//
//     // Forward method returns the transformed tensor and the log-determinant of the Jacobian
//     std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& x, const torch::Tensor& cond) {
//         return impl_->forward(x, cond);
//     }
// };
//
// // The implementation struct
// struct cIncFlowImpl : torch::nn::Module {
//     int64_t num_features_;          // Number of features in the input x (e.g., channels)
//     int64_t cond_embedding_dim_;    // Dimensionality of the conditioning input vector
//     int64_t hidden_dim_cond_net_;   // Hidden dimension for the conditioning network
//
//     // Conditioning network layers (MLP to produce scale and bias)
//     torch::nn::Linear fc1_cond_{nullptr};
//     torch::nn::Linear fc2_cond_{nullptr};
//     // No explicit activation after fc1_cond to allow more direct mapping,
//     // or you could add one like ReLU.
//     // The output of fc2_cond will be split into log_scale and bias.
//
//     cIncFlowImpl(int64_t num_features, int64_t cond_embedding_dim, int64_t hidden_dim_cond_net = 0) // hidden_dim can be 0 for direct map
//         : num_features_(num_features),
//           cond_embedding_dim_(cond_embedding_dim),
//           hidden_dim_cond_net_(hidden_dim_cond_net) {
//
//         TORCH_CHECK(num_features > 0, "num_features must be positive.");
//         TORCH_CHECK(cond_embedding_dim > 0, "cond_embedding_dim must be positive.");
//
//         if (hidden_dim_cond_net_ <= 0) { // Direct mapping from cond to scale/bias
//              hidden_dim_cond_net_ = cond_embedding_dim_; // Use cond_embedding_dim as input to fc2
//              fc2_cond_ = torch::nn::Linear(hidden_dim_cond_net_, 2 * num_features_);
//              register_module("fc2_cond", fc2_cond_);
//         } else { // Use a hidden layer
//             fc1_cond_ = torch::nn::Linear(cond_embedding_dim_, hidden_dim_cond_net_);
//             fc2_cond_ = torch::nn::Linear(hidden_dim_cond_net_, 2 * num_features_); // Output for log_scale and bias
//             register_module("fc1_cond", fc1_cond_);
//             register_module("fc2_cond", fc2_cond_);
//         }
//     }
//
//     std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& x, const torch::Tensor& cond) {
//         // x: input tensor, e.g., (N, C) or (N, C, H, W) or (N, C, L)
//         // cond: conditioning tensor, e.g., (N, cond_embedding_dim_)
//
//         TORCH_CHECK(x.size(0) == cond.size(0), "Batch size of x and cond must match. Got x: ", x.size(0), ", cond: ", cond.size(0));
//         TORCH_CHECK(x.size(1) == num_features_, "Input x channels (dim 1) must match num_features. Expected: ", num_features_, ", got: ", x.size(1));
//         TORCH_CHECK(cond.dim() == 2 && cond.size(1) == cond_embedding_dim_,
//                     "Conditioning input 'cond' must be 2D with shape (N, cond_embedding_dim). Got shape: ", cond.sizes());
//
//         // --- 1. Compute scale and bias from conditioning input ---
//         torch::Tensor cond_processed = cond;
//         if (fc1_cond_) { // If hidden layer exists
//             cond_processed = fc1_cond_->forward(cond_processed);
//             cond_processed = torch::relu(cond_processed); // Common to add activation
//         }
//         torch::Tensor log_scale_and_bias = fc2_cond_->forward(cond_processed); // (N, 2 * num_features_)
//
//         // Split into log_scale and bias
//         auto chunks = torch::chunk(log_scale_and_bias, 2, /*dim=*/1);
//         torch::Tensor log_scale_params = chunks[0]; // (N, num_features_)
//         torch::Tensor bias_params = chunks[1];      // (N, num_features_)
//
//         // Typically, the scale parameter is made positive, e.g., by exp(log_scale)
//         // To prevent scale from becoming too large or small, sometimes tanh is used on log_scale_params.
//         // For simplicity, using exp directly. Add a constant for stability if desired e.g. torch::exp(log_scale_params + 2.0)
//         torch::Tensor scale = torch::exp(log_scale_params); // (N, num_features_)
//
//         // --- 2. Reshape scale and bias for broadcasting with x ---
//         // We want scale and bias to be (N, C, 1, 1, ...) to multiply with x (N, C, D1, D2, ...)
//         std::vector<int64_t> view_shape;
//         view_shape.push_back(x.size(0));      // N
//         view_shape.push_back(num_features_);  // C
//         for (int64_t i = 2; i < x.dim(); ++i) {
//             view_shape.push_back(1);
//         }
//         torch::Tensor scale_reshaped = scale.view(view_shape);
//         torch::Tensor bias_reshaped = bias_params.view(view_shape);
//
//         // --- 3. Apply affine transformation ---
//         torch::Tensor y = scale_reshaped * x + bias_reshaped;
//
//         // --- 4. Compute log-determinant of the Jacobian ---
//         // For y_i = s_i * x_i + b_i, the Jacobian is diagonal with entries s_i.
//         // The log-determinant is sum(log|s_i|) over all transformed dimensions.
//         // log_scale = log(exp(log_scale_params)) = log_scale_params
//         // We need to sum log_scale_params across channels and multiply by spatial/sequential dimensions.
//         torch::Tensor log_det_jacobian = log_scale_params.sum(/*dim=*/1); // Sum over C, result (N,)
//
//         // If x has spatial/sequential dimensions (D1, D2, ...), multiply by their product.
//         for (int64_t i = 2; i < x.dim(); ++i) {
//             log_det_jacobian = log_det_jacobian * x.size(i);
//         }
//         // log_det_jacobian is now of shape (N,)
//
//         return std::make_tuple(y, log_det_jacobian);
//     }
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "cIncFlow(num_features=" << num_features_
//                << ", cond_embedding_dim=" << cond_embedding_dim_
//                << ", hidden_dim_cond_net=" << (fc1_cond_ ? std::to_string(hidden_dim_cond_net_) : "0 (direct map)")
//                << ")";
//     }
// };
// TORCH_MODULE(cIncFlow);
//
//
// // --- Example Usage ---
// int main() {
//     torch::manual_seed(0);
//
//     int64_t num_features = 3;       // Channels in x
//     int64_t cond_embedding_dim = 10; // Dimension of conditioning vector
//     int64_t N = 4;                  // Batch size
//
//     // --- Test Case 1: 2D input x (N, C), simple conditioning network ---
//     std::cout << "--- Test Case 1: 2D input x (N, C), no hidden layer in cond_net ---" << std::endl;
//     cIncFlow cflow_module1(num_features, cond_embedding_dim, /*hidden_dim_cond_net=*/0);
//     // std::cout << cflow_module1 << std::endl;
//
//     torch::Tensor x1 = torch::randn({N, num_features});
//     torch::Tensor cond1 = torch::randn({N, cond_embedding_dim});
//     std::cout << "Input x1 shape: " << x1.sizes() << std::endl;
//     std::cout << "Cond cond1 shape: " << cond1.sizes() << std::endl;
//
//     auto [y1, log_det_j1] = cflow_module1->forward(x1, cond1);
//
//     std::cout << "Output y1 shape: " << y1.sizes() << std::endl;
//     std::cout << "Log_det_jacobian1 shape: " << log_det_j1.sizes() << " (expected N)" << std::endl;
//     std::cout << "Log_det_j1 example value: " << log_det_j1[0].item<double>() << std::endl;
//     TORCH_CHECK(y1.sizes() == x1.sizes(), "Output y1 shape mismatch!");
//     TORCH_CHECK(log_det_j1.size(0) == N, "Log_det_j1 batch size mismatch!");
//
//     // --- Test Case 2: 4D input x (N, C, H, W), conditioning network with hidden layer ---
//     std::cout << "\n--- Test Case 2: 4D input x (N, C, H, W), with hidden layer in cond_net ---" << std::endl;
//     int64_t H = 8, W = 8;
//     int64_t hidden_cond = 32;
//     cIncFlow cflow_module2(num_features, cond_embedding_dim, hidden_cond);
//     // std::cout << cflow_module2 << std::endl;
//
//     torch::Tensor x2 = torch::randn({N, num_features, H, W});
//     torch::Tensor cond2 = torch::randn({N, cond_embedding_dim});
//     std::cout << "Input x2 shape: " << x2.sizes() << std::endl;
//     std::cout << "Cond cond2 shape: " << cond2.sizes() << std::endl;
//
//     auto [y2, log_det_j2] = cflow_module2->forward(x2, cond2);
//
//     std::cout << "Output y2 shape: " << y2.sizes() << std::endl;
//     std::cout << "Log_det_jacobian2 shape: " << log_det_j2.sizes() << " (expected N)" << std::endl;
//     std::cout << "Log_det_j2 example value for first batch item: " << log_det_j2[0].item<double>() << std::endl;
//     // Expected log_det_j contribution from spatial dims: H*W
//     // e.g. if sum of log_scales for channel 0 is S0, then contrib is S0*H*W
//     TORCH_CHECK(y2.sizes() == x2.sizes(), "Output y2 shape mismatch!");
//     TORCH_CHECK(log_det_j2.size(0) == N, "Log_det_j2 batch size mismatch!");
//
//
//     // --- Test Case 3: Check backward pass (requires gradients) ---
//     std::cout << "\n--- Test Case 3: Backward pass check ---" << std::endl;
//     cIncFlow cflow_module3(num_features, cond_embedding_dim, hidden_cond);
//     cflow_module3->train(); // Ensure parameters have requires_grad=true
//
//     torch::Tensor x3 = torch::randn({N, num_features, H, W}, torch::requires_grad());
//     torch::Tensor cond3 = torch::randn({N, cond_embedding_dim}, torch::requires_grad()); // Cond can also have grad
//
//     auto [y3, log_det_j3] = cflow_module3->forward(x3, cond3);
//     torch::Tensor loss = y3.mean() + log_det_j3.mean(); // Example loss
//     loss.backward();
//
//     bool grad_exists_x3 = x3.grad().defined() && x3.grad().abs().sum().item<double>() > 0;
//     bool grad_exists_cond3 = cond3.grad().defined() && cond3.grad().abs().sum().item<double>() > 0;
//     bool grad_exists_fc2_weight = cflow_module3->fc2_cond_->weight.grad().defined() &&
//                                   cflow_module3->fc2_cond_->weight.grad().abs().sum().item<double>() > 0;
//
//     std::cout << "Gradient exists for x3: " << (grad_exists_x3 ? "true" : "false") << std::endl;
//     std::cout << "Gradient exists for cond3: " << (grad_exists_cond3 ? "true" : "false") << std::endl;
//     std::cout << "Gradient exists for fc2_cond.weight: " << (grad_exists_fc2_weight ? "true" : "false") << std::endl;
//
//     TORCH_CHECK(grad_exists_x3, "No gradient for x3!");
//     TORCH_CHECK(grad_exists_cond3, "No gradient for cond3!");
//     TORCH_CHECK(grad_exists_fc2_weight, "No gradient for fc2_cond.weight!");
//
//
//     std::cout << "\ncIncFlow tests finished." << std::endl;
//     return 0;
// }


namespace xt::norm
{
    CINCFlow::CINCFlow(int64_t num_features, int64_t cond_embedding_dim, int64_t hidden_dim_cond_net)
    // hidden_dim can be 0 for direct map
        : num_features_(num_features),
          cond_embedding_dim_(cond_embedding_dim),
          hidden_dim_cond_net_(hidden_dim_cond_net)
    {
        TORCH_CHECK(num_features > 0, "num_features must be positive.");
        TORCH_CHECK(cond_embedding_dim > 0, "cond_embedding_dim must be positive.");

        if (hidden_dim_cond_net_ <= 0)
        {
            // Direct mapping from cond to scale/bias
            hidden_dim_cond_net_ = cond_embedding_dim_; // Use cond_embedding_dim as input to fc2
            fc2_cond_ = torch::nn::Linear(hidden_dim_cond_net_, 2 * num_features_);
            register_module("fc2_cond", fc2_cond_);
        }
        else
        {
            // Use a hidden layer
            fc1_cond_ = torch::nn::Linear(cond_embedding_dim_, hidden_dim_cond_net_);
            fc2_cond_ = torch::nn::Linear(hidden_dim_cond_net_, 2 * num_features_); // Output for log_scale and bias
            register_module("fc1_cond", fc1_cond_);
            register_module("fc2_cond", fc2_cond_);
        }
    }

    auto CINCFlow::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        vector<std::any> tensors_ = tensors;
        auto x = std::any_cast<torch::Tensor>(tensors_[0]);
        auto cond = std::any_cast<torch::Tensor>(tensors_[1]);

        // x: input tensor, e.g., (N, C) or (N, C, H, W) or (N, C, L)
        // cond: conditioning tensor, e.g., (N, cond_embedding_dim_)

        TORCH_CHECK(x.size(0) == cond.size(0), "Batch size of x and cond must match. Got x: ", x.size(0), ", cond: ",
                    cond.size(0));
        TORCH_CHECK(x.size(1) == num_features_, "Input x channels (dim 1) must match num_features. Expected: ",
                    num_features_, ", got: ", x.size(1));
        TORCH_CHECK(cond.dim() == 2 && cond.size(1) == cond_embedding_dim_,
                    "Conditioning input 'cond' must be 2D with shape (N, cond_embedding_dim). Got shape: ",
                    cond.sizes());

        // --- 1. Compute scale and bias from conditioning input ---
        torch::Tensor cond_processed = cond;
        if (fc1_cond_)
        {
            // If hidden layer exists
            cond_processed = fc1_cond_->forward(cond_processed);
            cond_processed = torch::relu(cond_processed); // Common to add activation
        }
        torch::Tensor log_scale_and_bias = fc2_cond_->forward(cond_processed); // (N, 2 * num_features_)

        // Split into log_scale and bias
        auto chunks = torch::chunk(log_scale_and_bias, 2, /*dim=*/1);
        torch::Tensor log_scale_params = chunks[0]; // (N, num_features_)
        torch::Tensor bias_params = chunks[1]; // (N, num_features_)

        // Typically, the scale parameter is made positive, e.g., by exp(log_scale)
        // To prevent scale from becoming too large or small, sometimes tanh is used on log_scale_params.
        // For simplicity, using exp directly. Add a constant for stability if desired e.g. torch::exp(log_scale_params + 2.0)
        torch::Tensor scale = torch::exp(log_scale_params); // (N, num_features_)

        // --- 2. Reshape scale and bias for broadcasting with x ---
        // We want scale and bias to be (N, C, 1, 1, ...) to multiply with x (N, C, D1, D2, ...)
        std::vector<int64_t> view_shape;
        view_shape.push_back(x.size(0)); // N
        view_shape.push_back(num_features_); // C
        for (int64_t i = 2; i < x.dim(); ++i)
        {
            view_shape.push_back(1);
        }
        torch::Tensor scale_reshaped = scale.view(view_shape);
        torch::Tensor bias_reshaped = bias_params.view(view_shape);

        // --- 3. Apply affine transformation ---
        torch::Tensor y = scale_reshaped * x + bias_reshaped;

        // --- 4. Compute log-determinant of the Jacobian ---
        // For y_i = s_i * x_i + b_i, the Jacobian is diagonal with entries s_i.
        // The log-determinant is sum(log|s_i|) over all transformed dimensions.
        // log_scale = log(exp(log_scale_params)) = log_scale_params
        // We need to sum log_scale_params across channels and multiply by spatial/sequential dimensions.
        torch::Tensor log_det_jacobian = log_scale_params.sum(/*dim=*/1); // Sum over C, result (N,)

        // If x has spatial/sequential dimensions (D1, D2, ...), multiply by their product.
        for (int64_t i = 2; i < x.dim(); ++i)
        {
            log_det_jacobian = log_det_jacobian * x.size(i);
        }
        // log_det_jacobian is now of shape (N,)

        return std::make_tuple(y, log_det_jacobian);
    }
}
