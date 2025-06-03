#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>
#include <any> // For potential future use in passing more data to callbacks

// Forward declaration to avoid circular dependency
namespace xt { class Trainer; }

namespace xt {

    // Struct to hold metrics/state passed to callbacks, can be expanded
    struct EpochEndState {
        int epoch;
        double train_loss;
        double val_loss = -1.0; // -1.0 if no validation
        // Add other metrics here, e.g., std::map<std::string, double> custom_metrics;
    };

    struct BatchEndState {
        int epoch;
        int batch_idx;
        torch::Tensor loss;
        torch::Tensor output;
        torch::Tensor target;
        size_t num_samples;
        // Add other metrics here
    };


    class Callback {
    public:
        virtual ~Callback() = default;

        // Called by Trainer to give callback access to trainer's state (if needed)
        // Non-owning pointer. Trainer's lifetime must exceed callback's or be managed carefully.
        virtual void set_trainer(Trainer* trainer) { trainer_ = trainer; }

        virtual void on_train_begin() {}
        virtual void on_train_end() {}

        virtual void on_epoch_begin(int epoch) {}
        // epoch_state contains losses and potentially other metrics
        virtual void on_epoch_end(const EpochEndState& epoch_state) {}

        virtual void on_batch_begin(int epoch, int batch_idx) {}
        // batch_state contains per-batch info
        virtual void on_batch_end(const BatchEndState& batch_state) {}

    protected:
        Trainer* trainer_ = nullptr; // Access to the trainer instance
    };

} // namespace xt