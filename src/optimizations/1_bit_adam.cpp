// #include "include/optimizations/1_bit_adam.h"
//
// namespace xt::optimizations
// {
//     OneBitAdam::OneBitAdam(std::vector<torch::Tensor> parameters, double lr, double momentum)
//         : torch::optim::Optimizer(std::move(parameters)), lr_(lr), momentum_(momentum)
//     {
//         velocities_.resize(param_groups()[0].params().size());
//         for (size_t i = 0; i < velocities_.size(); ++i)
//         {
//             velocities_[i] = torch::zeros_like(param_groups()[0].params()[i]);
//         }
//     }
// }
