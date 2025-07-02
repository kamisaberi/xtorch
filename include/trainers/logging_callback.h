#pragma once

#include "callback.h" // Your base xt::Callback header
#include <iostream>
#include <string>
#include <chrono>    // For timing
#include <iomanip>   // For std::fixed, std::setprecision

namespace xt {

    class LoggingCallback : public Callback {
    public:
        // Constructor
        explicit LoggingCallback(const std::string& log_prefix = "[Trainer]",
                                 int log_batch_interval = 50, // Log every 50 batches
                                 bool log_time = true);

        // Overridden callback methods
        void on_train_begin() override;
        void on_train_end() override;

        void on_epoch_begin(int epoch) override;
        void on_epoch_end(const EpochEndState& state) override;

        void on_batch_begin(int epoch, int batch_idx) override;
        void on_batch_end(const BatchEndState& state) override;

    private:
        std::string prefix_;
        int log_batch_interval_;
        bool log_time_;

        // For timing
        std::chrono::steady_clock::time_point train_start_time_;
        std::chrono::steady_clock::time_point epoch_start_time_;
        std::chrono::steady_clock::time_point batch_start_time_; // Could be used for per-batch timing if very verbose
    };

} // namespace xt