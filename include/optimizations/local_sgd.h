// #ifndef LOCAL_SGD_OPTIMIZER_HPP
// #define LOCAL_SGD_OPTIMIZER_HPP
//
// #include <torch/torch.h>
// #include <torch/serialize/archive.h>
// #include <c10d/ProcessGroup.hpp> // For distributed communication
//
// #include <vector>
// #include <memory>
// #include <functional>
//
// // --- Options for the inner SGD optimizer ---
// struct InnerSGDOptions : torch::optim::OptimizerOptions {
//     explicit InnerSGDOptions(double learning_rate = 0.1)
//         : torch::optim::OptimizerOptions() {
//         this->lr(learning_rate);
//     }
//     TORCH_ARG(double, momentum) = 0.9;
//     TORCH_ARG(double, weight_decay) = 1e-4;
// };
//
// // --- Main Options for LocalSGD ---
// struct LocalSGDOptions {
//     long sync_frequency; // The number of local steps (H) before communication
//     c10d::ProcessGroup* dist_process_group; // Pointer to the distributed process group
//     InnerSGDOptions inner_optimizer_options;
//
//     LocalSGDOptions(
//         long _sync_frequency,
//         c10d::ProcessGroup* _pg,
//         InnerSGDOptions _inner_opts = InnerSGDOptions())
//         : sync_frequency(_sync_frequency),
//           dist_process_group(_pg),
//           inner_optimizer_options(std::move(_inner_opts)) {
//         TORCH_CHECK(sync_frequency > 0, "Synchronization frequency must be positive.");
//         TORCH_CHECK(dist_process_group != nullptr, "A valid process group must be provided.");
//     }
// };
//
// // --- LocalSGD Optimizer Wrapper Class ---
// class LocalSGD {
// public:
//     // Does not inherit from torch::optim::Optimizer
//     LocalSGD(std::vector<torch::Tensor> params, LocalSGDOptions options);
//
//     // The step function takes no arguments, as it's called after loss.backward()
//     void step();
//     void zero_grad();
//
// private:
//     std::vector<torch::Tensor> params_;
//     LocalSGDOptions options_;
//     std::unique_ptr<torch::optim::SGD> inner_optimizer_;
//     long local_step_count_ = 0;
// };
//
// #endif // LOCAL_SGD_OPTIMIZER_HPP