// #ifndef DISTRIBUTED_SHAMPOO_HPP
// #define DISTRIBUTED_SHAMPOO_HPP
//
// #include <torch/torch.h>
// #include <torch/serialize/archive.h>
// // #include <c10/core/IValue.h>
// // #include <c10d/ProcessGroup.hpp> // For distributed communication
//
// #include <cmath>
// #include <vector>
// #include <memory>
// #include <string>
// #include <cstdint>
//
// // --- Options for DistributedShampoo ---
// struct DistributedShampooOptions :public torch::optim::OptimizerOptions {
//     double  lr;
//     explicit DistributedShampooOptions(double learning_rate = 1.0) // Grafting makes lr=1.0 a good default
//         : torch::optim::OptimizerOptions() {
//         this->lr=learning_rate;
//     }
//
//     TORCH_ARG(double, beta1) = 0.9;         // For momentum (Adam part)
//     TORCH_ARG(double, beta2) = 0.999;       // For Shampoo statistics (preconditioner)
//     TORCH_ARG(double, eps) = 1e-6;          // Damping for matrix inverse root
//     TORCH_ARG(double, weight_decay) = 1e-4; // Weight decay
//     TORCH_ARG(int, root_order) = 4;         // p-th root for preconditioner, 4 is common
//     TORCH_ARG(long, precondition_frequency) = 100; // How often to compute matrix inverse roots
//     TORCH_ARG(long, start_preconditioning_step) = 250; // Start preconditioning after this many steps
//     TORCH_ARG(int, block_size) = 128; // Block-diagonal preconditioning for dimensions larger than this
//
//     // Grafting replaces Shampoo's update magnitude with a simpler optimizer's magnitude. Essential for stability.
//     // Options: "SGD", "ADAM", "NONE"
//     TORCH_ARG(std::string, grafting_type) = "SGD";
//
//     // Pointer to the distributed process group. MUST be set for distributed training.
//     c10d::ProcessGroup* dist_process_group = nullptr;
//
//     void serialize(torch::serialize::OutputArchive& archive) const override;
//     void deserialize(torch::serialize::InputArchive& archive) override;
//     std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
// };
//
// // --- Parameter State for DistributedShampoo ---
// struct ShampooParamState : torch::optim::OptimizerParamState {
//     TORCH_ARG(torch::Tensor, step);
//     TORCH_ARG(torch::Tensor, momentum); // m_t
//
//     // For Shampoo preconditioning
//     std::vector<torch::Tensor> preconditioners; // L, R, etc. (statistics)
//     std::vector<torch::Tensor> inv_root_preconditioners; // L^{-1/p}, R^{-1/p}, etc.
//
//     // For grafting
//     TORCH_ARG(torch::Tensor, grafted_norm);
//
//     ShampooParamState() = default;
//     void serialize(torch::serialize::OutputArchive& archive) const override;
//     void deserialize(torch::serialize::InputArchive& archive) ;
//     std::unique_ptr<OptimizerParamState> clone() const override;
// };
//
// // --- DistributedShampoo Optimizer Class ---
// class DistributedShampoo : public torch::optim::Optimizer {
// public:
//     DistributedShampoo(std::vector<torch::Tensor> params, DistributedShampooOptions options);
//
//     using LossClosure = std::function<torch::Tensor()>;
//     torch::Tensor step(LossClosure closure = nullptr) override;
//     void save(torch::serialize::OutputArchive& archive) const override;
//     void load(torch::serialize::InputArchive& archive) override;
//
// protected:
//     std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() override;
//
// private:
//     // Helper functions
//     void _fallback_to_adam(
//         torch::Tensor& param,
//         const torch::Tensor& grad,
//         ShampooParamState& state,
//         const DistributedShampooOptions& options);
//
//     torch::Tensor _compute_grafted_norm(
//         const torch::Tensor& grad,
//         ShampooParamState& state,
//         const DistributedShampooOptions& options);
//
//     torch::Tensor _compute_matrix_inverse_root(
//         const torch::Tensor& matrix,
//         int root_order,
//         double epsilon);
// };
//
// #endif // DISTRIBUTED_SHAMPOO_HPP