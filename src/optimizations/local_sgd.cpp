// #include "local_sgd.hpp"
// #include <stdexcept>
//
// // --- LocalSGD Implementation ---
//
// LocalSGD::LocalSGD(std::vector<torch::Tensor> params, LocalSGDOptions options)
//     : params_(std::move(params)),
//       options_(std::move(options)) {
//
//     // Create the inner SGD optimizer that will handle the local parameter updates
//     inner_optimizer_ = std::make_unique<torch::optim::SGD>(
//         params_,
//         torch::optim::SGDOptions(options_.inner_optimizer_options.lr())
//             .momentum(options_.inner_optimizer_options.momentum())
//             .weight_decay(options_.inner_optimizer_options.weight_decay())
//     );
// }
//
// void LocalSGD::step() {
//     // 1. Perform a standard local step using the inner optimizer
//     inner_optimizer_->step();
//
//     // 2. Increment the local step counter
//     local_step_count_++;
//
//     // 3. Check if it's time to synchronize with other workers
//     if (local_step_count_ % options_.sync_frequency == 0) {
//         // --- Synchronization Step ---
//
//         // This vector will hold the asynchronous work handles for all_reduce
//         std::vector<c10::intrusive_ptr<c10d::Work>> works;
//
//         // Iterate through all parameters and start an all_reduce operation for each
//         for (auto& p : params_) {
//             // We want to average the parameters, so we all_reduce the sum
//             // and then divide by the world size.
//             // The operation is in-place on the tensor `p.data()`.
//             works.push_back(
//                 options_.dist_process_group->allreduce({p.data()}, c10d::AllreduceOptions{})
//             );
//         }
//
//         // Wait for all all_reduce operations to complete
//         for (auto& work : works) {
//             work->wait();
//         }
//
//         // Now that we have the sum, divide by the world size to get the average
//         int world_size = options_.dist_process_group->getSize();
//         for (auto& p : params_) {
//             p.data().div_(world_size);
//         }
//
//         // Optional: Reset the inner optimizer's state (e.g., momentum buffers)
//         // This is a key design choice in Local SGD variants. Resetting momentum
//         // after averaging is a common strategy to prevent stale momentum from
//         // pulling the averaged model in a strange direction.
//         inner_optimizer_ = std::make_unique<torch::optim::SGD>(
//             params_,
//             torch::optim::SGDOptions(options_.inner_optimizer_options.lr())
//                 .momentum(options_.inner_optimizer_options.momentum())
//                 .weight_decay(options_.inner_optimizer_options.weight_decay())
//         );
//
//         // Alternatively, one could manually iterate through `inner_optimizer_->state()`
//         // and call `.zero_()` on the momentum buffers, but re-creation is simpler.
//     }
// }
//
// void LocalSGD::zero_grad() {
//     // Delegate zero_grad to the inner optimizer
//     inner_optimizer_->zero_grad();
// }