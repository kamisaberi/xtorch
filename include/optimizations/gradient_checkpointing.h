// #ifndef GRADIENT_CHECKPOINTING_OPTIMIZER_HPP
// #define GRADIENT_CHECKPOINTING_OPTIMIZER_HPP
//
// #include <torch/torch.h>
// #include <torch/utils.h> // For torch::utils::checkpoint
//
// #include <vector>
// #include <memory>
// #include <functional>
//
// // --- Main Options for GradientCheckpointingOptimizer ---
// // We use the standard AdamOptions directly to avoid errors.
// struct GradientCheckpointingOptions {
//     std::shared_ptr<torch::nn::Module> model;
//     std::function<torch::Tensor(torch::Tensor, torch::Tensor)> loss_fn;
//     std::vector<std::shared_ptr<torch::nn::Module>> checkpointed_modules;
//     torch::optim::AdamOptions inner_optimizer_options;
//
//     GradientCheckpointingOptions(
//         std::shared_ptr<torch::nn::Module> _model,
//         std::function<torch::Tensor(torch::Tensor, torch::Tensor)> _loss_fn,
//         std::vector<std::shared_ptr<torch::nn::Module>> _checkpointed_modules,
//         torch::optim::AdamOptions _inner_opts) // Pass the real AdamOptions
//         : model(std::move(_model)),
//           loss_fn(std::move(_loss_fn)),
//           checkpointed_modules(std::move(_checkpointed_modules)),
//           inner_optimizer_options(std::move(_inner_opts)) {}
// };
//
//
// // --- GradientCheckpointingOptimizer Class ---
// class GradientCheckpointingOptimizer {
// public:
//     // This optimizer does not inherit from torch::optim::Optimizer
//     GradientCheckpointingOptimizer(GradientCheckpointingOptions options);
//
//     // The step function needs data to perform the forward/backward pass.
//     torch::Tensor step(const torch::Tensor& input, const torch::Tensor& target);
//
//     // Standard optimizer methods
//     void zero_grad();
//
// private:
//     std::shared_ptr<torch::nn::Module> model_;
//     std::vector<std::shared_ptr<torch::nn::Module>> checkpointed_modules_;
//     std::function<torch::Tensor(torch::Tensor, torch::Tensor)> loss_fn_;
//     std::unique_ptr<torch::optim::Adam> inner_optimizer_;
// };
//
// #endif // GRADIENT_CHECKPOINTING_OPTIMIZER_HPP