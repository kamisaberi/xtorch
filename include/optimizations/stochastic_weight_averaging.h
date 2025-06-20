#pragma once


#include "common.h"
// --- StochasticWeightAveraging Wrapper Class ---
// This is a wrapper, not a direct optimizer. It contains a base optimizer.
class StochasticWeightAveraging {
public:
    // The constructor takes an existing, fully-configured optimizer.
    StochasticWeightAveraging(std::unique_ptr<torch::optim::Optimizer> base_optimizer,
                               int swa_start_step,
                               int swa_update_frequency = 1);

    // Standard optimizer methods that delegate to the base optimizer
    void zero_grad();
    torch::Tensor step(torch::optim::Optimizer::LossClosure closure = nullptr);

    // SWA-specific methods
    // Call this at the end of training to use the averaged weights
    void swap_swa_weights();

private:
    std::unique_ptr<torch::optim::Optimizer> base_optimizer_;
    std::vector<torch::Tensor> swa_params_; // Stores the running average of weights
    int64_t swa_start_step_;
    int64_t swa_update_frequency_;
    int64_t n_averaged_ = 0; // Number of models we have averaged so far
    int64_t global_step_ = 0;
};

