#include "include/trainers/trainer.h"

namespace xt {

    Trainer::Trainer()
            : max_epochs_(0),
              optimizer_(nullptr),
              loss_fn_(nullptr),
              checkpoint_enabled_(false),
              checkpoint_path_(""),
              checkpoint_interval_(0)
    {
        // Default constructor initializes members
    }

    Trainer& Trainer::set_max_epochs(int maxEpochs) {
        max_epochs_ = maxEpochs;
        return *this;  // Return reference to self for chaining
    }
    Trainer& Trainer::set_optimizer(torch::optim::Optimizer *optimizer) {
        optimizer_ = optimizer;  // Take ownership or share
        return *this;
    }

    Trainer& Trainer::set_loss_fn(std::function<torch::Tensor(torch::Tensor, torch::Tensor)> lossFn) {
        loss_fn_ = lossFn;  // Take ownership or share
        return *this;
    }
    Trainer& Trainer::enable_checkpoint(const std::string& path, int interval) {
        checkpoint_path_ = path;
        checkpoint_interval_ = interval;
        checkpoint_enabled_ = true;  // Enable checkpointing
        return *this;
    }

} // namespace xt

