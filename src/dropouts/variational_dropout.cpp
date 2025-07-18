#include <dropouts/variational_dropout.h>


// #include <torch/torch.h>
// #include <vector>
// #include <cmath>     // For std::log, std::exp, std::sqrt
// #include <ostream>   // For std::ostream
//
// // Helper to initialize log_alpha based on an initial effective "dropout rate p"
// // where alpha = p / (1-p). So, log_alpha = log(p / (1-p)).
// // This p is just for initialization intuition; alpha is the direct parameter for noise variance.
// double calculate_initial_log_alpha(double initial_effective_p, double epsilon = 1e-7) {
//     if (initial_effective_p < epsilon) initial_effective_p = epsilon;
//     if (initial_effective_p >= 1.0 - epsilon) initial_effective_p = 1.0 - epsilon - epsilon; // Ensure 1-p is not too small
//     double alpha = initial_effective_p / (1.0 - initial_effective_p);
//     return std::log(alpha + epsilon); // Add epsilon to avoid log(0) if alpha is near 0
// }
//
//
// struct VariationalDropoutImpl : torch::nn::Module {
//     // log_alpha is the learnable parameter. alpha = exp(log_alpha) is the variance
//     // of the N(0, alpha) noise added to the identity, so units are multiplied by N(1, alpha).
//     torch::Tensor log_alpha_;
//     double epsilon_ = 1e-8; // For numerical stability, esp. in sqrt
//
//     // alpha_shape: Determines if log_alpha is scalar or a tensor (e.g., per-feature).
//     // initial_dropout_rate_for_alpha_init: An intuitive "p" to initialize alpha = p/(1-p).
//     VariationalDropoutImpl(c10::IntArrayRef alpha_shape = {},
//                            double initial_dropout_rate_for_alpha_init = 0.05) {
//
//         double initial_log_alpha_val = calculate_initial_log_alpha(initial_dropout_rate_for_alpha_init);
//
//         torch::Tensor log_alpha_init_tensor;
//         if (alpha_shape.empty()) { // Scalar alpha
//             log_alpha_init_tensor = torch::tensor(initial_log_alpha_val, torch::kFloat32);
//         } else {
//             log_alpha_init_tensor = torch::full(alpha_shape, initial_log_alpha_val, torch::kFloat32);
//         }
//         log_alpha_ = register_parameter("log_alpha", log_alpha_init_tensor);
//     }
//
//     torch::Tensor forward(const torch::Tensor& input) {
//         if (!this->is_training()) {
//             // During evaluation, Variational Dropout (Kingma et al. version) acts as an identity.
//             // The learned alpha values are expected to have regularized the network.
//             return input;
//         }
//
//         torch::Tensor alpha = torch::exp(log_alpha_); // alpha = variance of N(0,alpha) noise
//
//         // Clamp alpha to prevent it from becoming too small (leading to NaN in sqrt) or excessively large.
//         // The lower bound for alpha (variance) should be positive.
//         alpha = torch::clamp_min(alpha, epsilon_);
//
//
//         // Handle broadcasting of alpha (e.g., for per-channel dropout).
//         torch::Tensor alpha_broadcastable = alpha;
//         if (alpha.dim() == 1 && input.dim() > 1 && alpha.size(0) == input.size(1) && input.size(1)>0) {
//             // Assuming alpha.size(0) is the channel dimension (dim 1 of input)
//             std::vector<int64_t> view_shape(input.dim(), 1L);
//             view_shape[1] = alpha.size(0); // e.g., [1, C, 1, 1] for NCHW input
//             alpha_broadcastable = alpha.view(view_shape);
//         }
//         // Else: rely on PyTorch's standard broadcasting if alpha is scalar or already matches.
//
//         // Generate standard Gaussian noise epsilon ~ N(0, 1)
//         torch::Tensor noise_std_normal = torch::randn_like(input);
//
//         // Apply noise: output = input * (1 + sqrt(alpha_broadcastable) * noise_std_normal)
//         // This is equivalent to input * N(1, alpha_broadcastable)
//         torch::Tensor scaled_noise = torch::sqrt(alpha_broadcastable) * noise_std_normal;
//         torch::Tensor output = input * (1.0 + scaled_noise);
//
//         return output;
//     }
//
//     // Method to calculate the KL divergence term for regularization.
//     // KL[q(w)||p(w)] approx sum_i -0.5 * log_alpha_i + const. (Simplified from Kingma et al. for this setup)
//     // Or, more directly from paper for p(theta_i)=N(0,1), q(theta_i)=N(mu_i, sigma_i^2=alpha_i*mu_i^2):
//     // For weight w_i, if dropout is multiplicative noise 1+eps_i*sqrt(alpha_i),
//     // and w_i is seen as fixed, this dropout affects activations.
//     // The KL term from Kingma's paper (Eq. 10) for q(epsilon_i) vs p(epsilon_i)
//     // where epsilon is standard Gaussian and we learn alpha (variance of activation noise):
//     // KL(q(W)|p(W)) = sum_units [ 0.5 * (alpha + mu^2 - 1 - log(alpha) ) ] if p(W)=N(0,I)
//     // If we're just learning alpha for multiplicative noise N(1, alpha) on activations,
//     // the regularization is often: C * sum_units ( log(1+1/alpha_unit) ) (from Concrete Dropout related ideas)
//     // or from Kingma et al. for activations: KL = sum_units ( -0.5 * log_alpha_unit ) + const
//     // For this module, let's provide a way to get alpha. The user must implement the correct KL term.
//     torch::Tensor get_alpha() const {
//         return torch::exp(log_alpha_);
//     }
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "VariationalDropout(alpha_shape=" << log_alpha_.sizes();
//         if (log_alpha_.numel() > 0) {
//              stream << ", current_alpha_mean_approx=" << torch::exp(log_alpha_.mean()).item<double>();
//         }
//         stream << ")";
//     }
// };
//
// TORCH_MODULE(VariationalDropout);
//
// /*
// // Example of how to use the VariationalDropout module:
// #include <iostream>
// #include <torch/optim.hpp> // For torch::optim::Adam
// #include <iomanip>          // For std::fixed, std::setprecision
//
// void run_variational_dropout_example() {
//     torch::manual_seed(0);
//     std::cout << std::fixed << std::setprecision(4);
//
//     // Initial effective "p" for alpha initialization. alpha = p/(1-p).
//     double initial_p_for_alpha = 0.1; // So initial alpha around 0.1/0.9 = 0.111
//
//     // 1. VariationalDropout with a scalar (global) learnable alpha
//     VariationalDropout global_vd_module({}, initial_p_for_alpha);
//     std::cout << "Global VariationalDropout Module: " << global_vd_module << std::endl;
//     std::cout << "Initial log_alpha: " << global_vd_module->log_alpha_.item<double>() << std::endl;
//     std::cout << "Initial alpha (variance): " << global_vd_module->get_alpha().item<double>() << std::endl;
//
//     torch::Tensor input_tensor = torch::ones({2, 4}); // Example input
//     global_vd_module->train(); // Set to training mode
//
//     // --- Simulate a training step ---
//     torch::optim::Adam optimizer(global_vd_module->parameters(), torch::optim::AdamOptions(1e-2));
//     optimizer.zero_grad();
//
//     torch::Tensor output_train = global_vd_module(input_tensor);
//
//     // Dummy main loss
//     torch::Tensor main_task_loss = torch::mse_loss(output_train, torch::ones_like(output_train) * 0.5);
//
//     // **IMPORTANT**: Calculate KL regularization term for Variational Dropout.
//     // This is a simplified placeholder. The actual KL term depends on the specific
//     // variational family and prior chosen. For Kingma et al.'s activation noise:
//     // KL ~ sum_i -0.5 * log_alpha_i (ignoring constants and terms related to weights mean).
//     // Let's use a proxy like: lambda * sum(alpha) or lambda * sum(-0.5 * log_alpha)
//     double lambda_kl = 1e-4;
//     torch::Tensor current_alpha = global_vd_module->get_alpha();
//     // A common form for regularization related to alpha (variance) might be:
//     torch::Tensor kl_regularization_loss = lambda_kl * torch::sum(-0.5 * global_vd_module->log_alpha_);
//     // Or, sometimes a penalty encouraging smaller alpha (less noise): lambda_kl * torch::sum(current_alpha);
//
//     torch::Tensor total_loss = main_task_loss + kl_regularization_loss;
//     total_loss.backward();
//     optimizer.step();
//
//     std::cout << "\n--- After one optimization step (Global VD) ---" << std::endl;
//     std::cout << "Input (all ones):\n" << input_tensor << std::endl;
//     std::cout << "Output (training):\n" << output_train << std::endl;
//     std::cout << "Main Task Loss: " << main_task_loss.item<double>() << std::endl;
//     std::cout << "KL Regularization Loss: " << kl_regularization_loss.item<double>() << std::endl;
//     std::cout << "Total Loss: " << total_loss.item<double>() << std::endl;
//     std::cout << "log_alpha after step: " << global_vd_module->log_alpha_.item<double>() << std::endl;
//     std::cout << "alpha (variance) after step: " << global_vd_module->get_alpha().item<double>() << std::endl;
//
//     global_vd_module->eval();
//     torch::Tensor output_eval = global_vd_module(input_tensor);
//     std::cout << "Output (evaluation):\n" << output_eval << std::endl; // Should be same as input
//     TORCH_CHECK(torch::allclose(input_tensor, output_eval), "Global VD eval mismatch");
//
//
//     // 2. VariationalDropout with per-feature learnable alpha
//     int num_features = 3;
//     VariationalDropout per_feature_vd_module({num_features}, 0.2); // p_shape {3}, initial_p for alpha = 0.2
//     std::cout << "\nPer-Feature VariationalDropout Module: " << per_feature_vd_module << std::endl;
//     std::cout << "Initial log_alphas (per-feature):\n" << per_feature_vd_module->log_alpha_ << std::endl;
//     std::cout << "Initial alphas (per-feature):\n" << per_feature_vd_module->get_alpha() << std::endl;
//
//     torch::Tensor feature_input = torch::randn({2, num_features}); // Batch=2, Features=3
//     per_feature_vd_module->train();
//     torch::Tensor feature_output_train = per_feature_vd_module(feature_input);
//     std::cout << "Feature Input:\n" << feature_input << std::endl;
//     std::cout << "Feature Output (train, per-feature alpha):\n" << feature_output_train << std::endl;
//     // Again, KL regularization would be calculated using per_feature_vd_module->log_alpha_ or get_alpha()
// }
//
// // int main() {
// //    run_variational_dropout_example();
// //    return 0;
// // }
// */
//


namespace xt::dropouts
{
    namespace
    {
        double calculate_initial_log_alpha(double initial_effective_p, double epsilon = 1e-7)
        {
            if (initial_effective_p < epsilon) initial_effective_p = epsilon;
            if (initial_effective_p >= 1.0 - epsilon) initial_effective_p = 1.0 - epsilon - epsilon;
            // Ensure 1-p is not too small
            double alpha = initial_effective_p / (1.0 - initial_effective_p);
            return std::log(alpha + epsilon); // Add epsilon to avoid log(0) if alpha is near 0
        }
    }

    VariationalDropout::VariationalDropout(c10::IntArrayRef alpha_shape, double initial_dropout_rate_for_alpha_init)
    {
        double initial_log_alpha_val = calculate_initial_log_alpha(initial_dropout_rate_for_alpha_init);

        torch::Tensor log_alpha_init_tensor;
        if (alpha_shape.empty())
        {
            // Scalar alpha
            log_alpha_init_tensor = torch::tensor(initial_log_alpha_val, torch::kFloat32);
        }
        else
        {
            log_alpha_init_tensor = torch::full(alpha_shape, initial_log_alpha_val, torch::kFloat32);
        }
        log_alpha_ = register_parameter("log_alpha", log_alpha_init_tensor);
    }

    auto VariationalDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        vector<std::any> tensors_ = tensors;
        auto input = std::any_cast<torch::Tensor>(tensors_[0]);


        if (!this->is_training())
        {
            // During evaluation, Variational Dropout (Kingma et al. version) acts as an identity.
            // The learned alpha values are expected to have regularized the network.
            return input;
        }

        torch::Tensor alpha = torch::exp(log_alpha_); // alpha = variance of N(0,alpha) noise

        // Clamp alpha to prevent it from becoming too small (leading to NaN in sqrt) or excessively large.
        // The lower bound for alpha (variance) should be positive.
        alpha = torch::clamp_min(alpha, epsilon_);


        // Handle broadcasting of alpha (e.g., for per-channel dropout).
        torch::Tensor alpha_broadcastable = alpha;
        if (alpha.dim() == 1 && input.dim() > 1 && alpha.size(0) == input.size(1) && input.size(1) > 0)
        {
            // Assuming alpha.size(0) is the channel dimension (dim 1 of input)
            std::vector<int64_t> view_shape(input.dim(), 1L);
            view_shape[1] = alpha.size(0); // e.g., [1, C, 1, 1] for NCHW input
            alpha_broadcastable = alpha.view(view_shape);
        }
        // Else: rely on PyTorch's standard broadcasting if alpha is scalar or already matches.

        // Generate standard Gaussian noise epsilon ~ N(0, 1)
        torch::Tensor noise_std_normal = torch::randn_like(input);

        // Apply noise: output = input * (1 + sqrt(alpha_broadcastable) * noise_std_normal)
        // This is equivalent to input * N(1, alpha_broadcastable)
        torch::Tensor scaled_noise = torch::sqrt(alpha_broadcastable) * noise_std_normal;
        torch::Tensor output = input * (1.0 + scaled_noise);

        return output;
    }
}
