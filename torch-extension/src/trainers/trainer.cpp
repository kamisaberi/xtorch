#include "../../include/trainers/trainer.h"


namespace xt {

    Trainer::Trainer()
        : maxEpochs_(0),
          optimizer_(nullptr),
          lossFn_(nullptr),
          checkpointEnabled_(false),
          checkpointPath_(""),
          checkpointInterval_(0)
    {
        // Default constructor initializes members
    }

    Trainer& Trainer::setMaxEpochs(int maxEpochs) {
        maxEpochs_ = maxEpochs;
        return *this;  // Return reference to self for chaining
    }

    Trainer& Trainer::setOptimizer(std::shared_ptr<Optimizer> optimizer) {
        optimizer_ = std::move(optimizer);  // Take ownership or share
        return *this;
    }

    Trainer& Trainer::setLossFn(std::shared_ptr<LossFunction> lossFn) {
        lossFn_ = std::move(lossFn);  // Take ownership or share
        return *this;
    }

    Trainer& Trainer::enableCheckpoint(const std::string& path, int interval) {
        checkpointPath_ = path;
        checkpointInterval_ = interval;
        checkpointEnabled_ = true;  // Enable checkpointing
        return *this;
    }

} // namespace xt