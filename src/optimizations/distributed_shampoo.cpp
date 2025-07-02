// #include "include/optimizations/distributed_shampoo.h"
// #include <stdexcept>
//
// // --- DistributedShampooOptions Methods ---
// void DistributedShampooOptions::serialize(torch::serialize::OutputArchive& archive) const {
//     archive.write("lr", this->lr());
//     archive.write("beta1", beta1());
//     archive.write("beta2", beta2());
//     archive.write("eps", eps());
//     archive.write("weight_decay", weight_decay());
//     archive.write("root_order", static_cast<int64_t>(root_order()));
//     archive.write("precondition_frequency", precondition_frequency());
//     archive.write("start_preconditioning_step", start_preconditioning_step());
//     archive.write("block_size", static_cast<int64_t>(block_size()));
//     archive.write("grafting_type", grafting_type());
//     // Note: dist_process_group is not serialized. It must be re-attached after loading.
// }
//
// void DistributedShampooOptions::deserialize(torch::serialize::InputArchive& archive) {
//     c10::IValue ivalue;
//     if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
//     if (archive.try_read("beta1", ivalue)) { beta1_ = ivalue.toDouble(); }
//     if (archive.try_read("beta2", ivalue)) { beta2_ = ivalue.toDouble(); }
//     if (archive.try_read("eps", ivalue)) { eps_ = ivalue.toDouble(); }
//     if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
//     if (archive.try_read("root_order", ivalue)) { root_order_ = ivalue.toInt(); }
//     if (archive.try_read("precondition_frequency", ivalue)) { precondition_frequency_ = ivalue.toInt(); }
//     if (archive.try_read("start_preconditioning_step", ivalue)) { start_preconditioning_step_ = ivalue.toInt(); }
//     if (archive.try_read("block_size", ivalue)) { block_size_ = ivalue.toInt(); }
//     if (archive.try_read("grafting_type", ivalue)) { grafting_type_ = ivalue.toStringRef(); }
// }
//
// std::unique_ptr<torch::optim::OptimizerOptions> DistributedShampooOptions::clone() const {
//     auto cloned = std::make_unique<DistributedShampooOptions>(this->lr());
//     cloned->beta1(beta1()).beta2(beta2()).eps(eps()).weight_decay(weight_decay())
//           .root_order(root_order()).precondition_frequency(precondition_frequency())
//           .start_preconditioning_step(start_preconditioning_step())
//           .block_size(block_size()).grafting_type(grafting_type());
//     cloned->dist_process_group = dist_process_group; // Copy pointer
//     return cloned;
// }
//
// // --- ShampooParamState Methods ---
// void ShampooParamState::serialize(torch::serialize::OutputArchive& archive) const {
//     archive.write("step", step(), true);
//     archive.write("momentum", momentum(), true);
//     if(grafted_norm().defined()) archive.write("grafted_norm", grafted_norm(), true);
//
//     archive.write("num_preconditioners", static_cast<int64_t>(preconditioners.size()));
//     for(size_t i=0; i<preconditioners.size(); ++i) {
//         archive.write("preconditioner_" + std::to_string(i), preconditioners[i], true);
//         archive.write("inv_root_preconditioner_" + std::to_string(i), inv_root_preconditioners[i], true);
//     }
// }
//
// void ShampooParamState::deserialize(torch::serialize::InputArchive& archive) {
//     at::Tensor temp_tensor;
//     if(archive.try_read("step", temp_tensor, true)) step_ = temp_tensor;
//     if(archive.try_read("momentum", temp_tensor, true)) momentum_ = temp_tensor;
//     if(archive.try_read("grafted_norm", temp_tensor, true)) grafted_norm_ = temp_tensor;
//
//     c10::IValue ivalue;
//     if (archive.try_read("num_preconditioners", ivalue)) {
//         int64_t num_preconditioners = ivalue.toInt();
//         preconditioners.resize(num_preconditioners);
//         inv_root_preconditioners.resize(num_preconditioners);
//         for (int64_t i = 0; i < num_preconditioners; ++i) {
//             archive.read("preconditioner_" + std::to_string(i), preconditioners[i], true);
//             archive.read("inv_root_preconditioner_" + std::to_string(i), inv_root_preconditioners[i], true);
//         }
//     }
// }
//
// std::unique_ptr<torch::optim::OptimizerParamState> ShampooParamState::clone() const {
//     auto cloned = std::make_unique<ShampooParamState>();
//     if (step().defined()) cloned->step(step().clone());
//     if (momentum().defined()) cloned->momentum(momentum().clone());
//     if (grafted_norm().defined()) cloned->grafted_norm(grafted_norm().clone());
//
//     for(const auto& p : preconditioners) cloned->preconditioners.push_back(p.clone());
//     for(const auto& p : inv_root_preconditioners) cloned->inv_root_preconditioners.push_back(p.clone());
//     return cloned;
// }
//
//
// // --- DistributedShampoo Implementation ---
// DistributedShampoo::DistributedShampoo(std::vector<torch::Tensor> params, DistributedShampooOptions options)
//     : torch::optim::Optimizer(std::move(params), std::make_unique<DistributedShampooOptions>(options)) {}
//
// torch::Tensor DistributedShampoo::step(LossClosure closure) {
//     torch::NoGradGuard no_grad;
//     torch::Tensor loss = {};
//     if (closure) { loss = closure(); }
//
//     auto& group_options = static_cast<DistributedShampooOptions&>(param_groups_[0].options());
//     ProcessGroup* process_group = group_options.dist_process_group;
//
//     for (auto& p : param_groups_[0].params()) {
//         if (!p.grad().defined()) { continue; }
//
//         auto grad = p.grad();
//         auto& state = static_cast<ShampooParamState&>(*state_.at(p.unsafeGetTensorImpl()));
//
//         // Initialize state
//         if (!state.step().defined()) {
//             state.step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
//             state.momentum(torch::zeros_like(p));
//         }
//
//         // Increment step
//         state.step(state.step() + 1.0);
//         double current_step_val = state.step().item<double>();
//
//         // Fallback for 1D tensors (biases) or small parameters
//         if (p.dim() <= 1) {
//             _fallback_to_adam(p, grad, state, group_options);
//             continue;
//         }
//
//         // --- Main Shampoo Logic ---
//
//         // 1. Update Momentum
//         state.momentum().mul_(group_options.beta1()).add_(grad, 1.0 - group_options.beta1());
//
//         // 2. Gather Statistics (Preconditioners)
//         std::vector<torch::Tensor> statistics;
//         std::vector<int64_t> shapes;
//
//         for (int dim = 0; dim < p.dim(); ++dim) {
//             shapes.push_back(p.size(dim));
//             int block_size = group_options.block_size();
//
//             if (p.size(dim) > block_size) {
//                 // Matricize gradient for this dimension
//                 auto grad_reshaped = grad.permute(dim, at::IntArrayRef( // Move current dim to front
//                     [&]{
//                         std::vector<int64_t> dims;
//                         for(int i=0; i<p.dim(); ++i) if (i != dim) dims.push_back(i);
//                         return dims;
//                     }()
//                 )).reshape({p.size(dim), -1});
//
//                 // Compute statistic: G G^T
//                 auto stat = grad_reshaped.matmul(grad_reshaped.t());
//                 statistics.push_back(stat);
//             }
//         }
//
//         // DISTRIBUTED: all_reduce the statistics
//         if (process_group) {
//             std::vector<c10::intrusive_ptr<c10d::Work>> works;
//             for (auto& stat : statistics) {
//                 works.push_back(process_group->allreduce({stat}));
//             }
//             for (auto& work : works) {
//                 work->wait();
//             }
//         }
//
//         // Update EMA of statistics
//         if (state.preconditioners.empty()) { // Initialize
//             for (const auto& stat : statistics) {
//                 state.preconditioners.push_back(torch::zeros_like(stat));
//                 state.inv_root_preconditioners.push_back(torch::eye(stat.size(0), stat.options()));
//             }
//         }
//
//         int stat_idx = 0;
//         for (int dim = 0; dim < p.dim(); ++dim) {
//             if (p.size(dim) > group_options.block_size()) {
//                 state.preconditioners[stat_idx].mul_(group_options.beta2()).add_(statistics[stat_idx], 1.0 - group_options.beta2());
//                 stat_idx++;
//             }
//         }
//
//         // 3. Compute Inverse Roots (infrequently)
//         if (current_step_val >= group_options.start_preconditioning_step() &&
//             static_cast<long>(current_step_val) % group_options.precondition_frequency() == 0) {
//
//             for (size_t i = 0; i < state.preconditioners.size(); ++i) {
//                 state.inv_root_preconditioners[i] = _compute_matrix_inverse_root(
//                     state.preconditioners[i], group_options.root_order(), group_options.eps()
//                 );
//             }
//         }
//
//         // 4. Compute Preconditioned Gradient
//         torch::Tensor preconditioned_grad = state.momentum().clone();
//         stat_idx = 0;
//         for(int dim=0; dim < p.dim(); ++dim) {
//              if (p.size(dim) > group_options.block_size()) {
//                  const auto& inv_root = state.inv_root_preconditioners[stat_idx];
//
//                  // Reshape to apply matrix multiplication
//                  auto grad_reshaped = preconditioned_grad.permute(dim, at::IntArrayRef(
//                     [&]{
//                         std::vector<int64_t> dims;
//                         for(int i=0; i<p.dim(); ++i) if (i != dim) dims.push_back(i);
//                         return dims;
//                     }()
//                  )).reshape({p.size(dim), -1});
//
//                  // Apply preconditioning
//                  auto precond_grad_reshaped = inv_root.matmul(grad_reshaped);
//
//                  // Reshape back to original permuted shape and then un-permute
//                  preconditioned_grad = precond_grad_reshaped.reshape(preconditioned_grad.permute(dim, at::IntArrayRef(
//                     [&]{
//                         std::vector<int64_t> dims;
//                         for(int i=0; i<p.dim(); ++i) if (i != dim) dims.push_back(i);
//                         return dims;
//                     }()
//                  )).sizes()).permute(at::IntArrayRef( // Invert the permutation
//                     [&]{
//                         std::vector<int64_t> inv_perm(p.dim());
//                         inv_perm[dim] = 0;
//                         int current = 1;
//                         for(int i=0; i<p.dim(); ++i) if(i != dim) inv_perm[i] = current++;
//                         return inv_perm;
//                     }()
//                  ));
//
//                  stat_idx++;
//              }
//         }
//
//         // 5. Grafting
//         torch::Tensor grafted_norm = _compute_grafted_norm(grad, state, group_options);
//         torch::Tensor shampoo_norm = preconditioned_grad.norm();
//
//         torch::Tensor final_update = preconditioned_grad;
//         if (shampoo_norm.item<double>() > 1e-10) {
//             final_update.mul_(grafted_norm / shampoo_norm);
//         }
//
//         // 6. Weight Decay & Final Parameter Update
//         if (group_options.weight_decay() > 0.0) {
//             p.data().mul_(1.0 - group_options.lr() * group_options.weight_decay());
//         }
//         p.data().add_(final_update, -group_options.lr());
//     }
//     return loss;
// }
//
// // --- Helper Functions ---
//
// void DistributedShampoo::_fallback_to_adam(
//     torch::Tensor& param,
//     const torch::Tensor& grad,
//     ShampooParamState& state,
//     const DistributedShampooOptions& options) {
//
//     // Simple Adam update for this parameter
//     state.momentum().mul_(options.beta1()).add_(grad, 1.0 - options.beta1());
//
//     // No v_t is stored, so this is Adam without the adaptive learning rate part.
//     // It's effectively momentum SGD with bias correction.
//     double bias_correction1 = 1.0 - std::pow(options.beta1(), state.step().item<double>());
//     torch::Tensor update = state.momentum() / bias_correction1;
//
//     if (options.weight_decay() > 0.0) {
//         param.data().mul_(1.0 - options.lr() * options.weight_decay());
//     }
//     param.data().add_(update, -options.lr());
// }
//
// torch::Tensor DistributedShampoo::_compute_grafted_norm(
//     const torch::Tensor& grad,
//     ShampooParamState& state,
//     const DistributedShampooOptions& options) {
//
//     if (options.grafting_type() == "SGD") {
//         return grad.norm();
//     }
//     if (options.grafting_type() == "ADAM") {
//         // For simplicity, we use SGD's norm here.
//         // A full Adam graft would require storing v_t.
//         return grad.norm();
//     }
//     return torch::tensor(1.0, grad.options()); // No grafting, norm is 1
// }
//
// torch::Tensor DistributedShampoo::_compute_matrix_inverse_root(
//     const torch::Tensor& matrix,
//     int root_order,
//     double epsilon) {
//
//     // Use symmetric eigendecomposition for stability
//     auto eigh_result = torch::linalg::eigh(matrix, "U");
//     auto eigenvalues = std::get<0>(eigh_result);
//     auto eigenvectors = std::get<1>(eigh_result);
//
//     // Damping and inverse root
//     torch::Tensor damped_eigenvalues = eigenvalues.clamp_min(epsilon).pow(-1.0 / static_cast<double>(root_order));
//
//     // Reconstruct: V @ D_inv_root @ V.T
//     return eigenvectors.matmul(
//         torch::diag(damped_eigenvalues).matmul(eigenvectors.t())
//     );
// }
//
// // --- Boilerplate Methods ---
// void DistributedShampoo::save(torch::serialize::OutputArchive& archive) const {
//     torch::optim::Optimizer::save(archive);
// }
//
// void DistributedShampoo::load(torch::serialize::InputArchive& archive) {
//     torch::optim::Optimizer::load(archive);
// }
//
// std::unique_ptr<torch::optim::OptimizerParamState> DistributedShampoo::make_param_state() {
//     return std::make_unique<ShampooParamState>();
// }