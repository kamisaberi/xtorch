// #include "include/optimizations/gradient_checkpointing.h"
// #include <stdexcept>
//
// #include "models/reinforcement_learning/a3c.h"
//
// // --- GradientCheckpointingOptimizer Implementation ---
//
// // DEFINITIVELY CORRECTED CONSTRUCTOR
// GradientCheckpointingOptimizer::GradientCheckpointingOptimizer(GradientCheckpointingOptions options)
//     : model_(options.model),
//       checkpointed_modules_(options.checkpointed_modules),
//       loss_fn_(options.loss_fn) {
//
//     TORCH_CHECK(model_ != nullptr, "A valid model must be provided.");
//     TORCH_CHECK(loss_fn_ != nullptr, "A valid loss function must be provided.");
//
//     // The options object already contains a fully-formed AdamOptions struct.
//     // We can pass it directly to the Adam constructor.
//     inner_optimizer_ = std::make_unique<torch::optim::Adam>(
//         model_->parameters(),
//         options.inner_optimizer_options
//     );
// }
//
// // The custom step function that orchestrates the entire checkpointed pass
// torch::Tensor GradientCheckpointingOptimizer::step(const torch::Tensor& input, const torch::Tensor& target) {
//     // This function requires gradients, so no NoGradGuard here.
//
//     torch::Tensor current_tensor = input;
//
//     auto sequential_model = dynamic_cast<torch::nn::SequentialImpl*>(model_.get());
//     TORCH_CHECK(sequential_model != nullptr, "This implementation requires the model to be a torch::nn::Sequential for simplicity.");
//
//     for (const auto& module : sequential_model->children()) {
//         bool should_checkpoint = false;
//         // Check if the current module is in the list of modules to checkpoint
//         for (const auto& chk_module : checkpointed_modules_) {
//             if (module.get() == chk_module.get()) {
//                 should_checkpoint = true;
//                 break;
//             }
//         }
//
//         if (should_checkpoint) {
//             auto fn = [module](torch::Tensor x) {
//                 return module->as<xt::Module>()->forward({x});
//             };
//             current_tensor = torch::utils::checkpoint(fn, current_tensor);
//         } else {
//             current_tensor = module->as<xt::Module>()->forward({current_tensor});
//         }
//     }
//
//     torch::Tensor output = current_tensor;
//
//     // Compute loss
//     torch::Tensor loss = loss_fn_(output, target);
//
//     // Perform backward pass
//     loss.backward();
//
//     // Step the inner optimizer, which has its own NoGradGuard
//     inner_optimizer_->step();
//
//     return loss;
// }
//
// void GradientCheckpointingOptimizer::zero_grad() {
//     inner_optimizer_->zero_grad();
// }