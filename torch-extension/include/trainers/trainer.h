#pragma once

#include <torch/torch.h>
#include <iostream>
#include <filesystem>

#include <memory>  // For std::shared_ptr
#include <string>  // For std::string

// Forward declarations (assume these classes exist elsewhere)
//class Optimizer;
//class LossFunction;

namespace xt {

    class Trainer {
    public:
        // Default constructor
        Trainer();

        // Setter methods with fluent interface
        Trainer& setMaxEpochs(int maxEpochs);
        Trainer& setOptimizer(std::shared_ptr<Optimizer> optimizer);
        Trainer& setLossFn(std::shared_ptr<LossFunction> lossFn);
        Trainer& enableCheckpoint(const std::string& path, int interval);

    private:
        int maxEpochs_;                         // Maximum number of training epochs
        std::shared_ptr<Optimizer> optimizer_;  // Optimizer object
        std::shared_ptr<LossFunction> lossFn_;  // Loss function object
        bool checkpointEnabled_;                // Flag for checkpointing status
        std::string checkpointPath_;            // Path for saving checkpoints
        int checkpointInterval_;                // Interval for checkpointing
    };

} // namespace xt



