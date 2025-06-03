#include "include/trainers/logging_callback.h"
#include "include/trainers/trainer.h" // To potentially access trainer->get_max_epochs(), etc. (optional)

namespace xt {

    LoggingCallback::LoggingCallback(const std::string& log_prefix, int log_batch_interval, bool log_time)
        : prefix_(log_prefix),
          log_batch_interval_(log_batch_interval > 0 ? log_batch_interval : 50), // Ensure positive
          log_time_(log_time) {}

    void LoggingCallback::on_train_begin() {
        if (log_time_) {
            train_start_time_ = std::chrono::steady_clock::now();
        }
        std::cout << prefix_ << " Training started." << std::endl;
        if (trainer_) { // Check if trainer pointer is set
            // You could access trainer_->get_max_epochs() here if needed, for example.
            // std::cout << prefix_ << " Max epochs: " << trainer_->get_max_epochs() << std::endl;
        }
    }

    void LoggingCallback::on_train_end() {
        std::cout << prefix_ << " Training finished. ";
        if (log_time_) {
            auto end_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - train_start_time_);
            std::cout << "Total training time: " << duration.count() << "s.";
        }
        std::cout << std::endl;
    }

    void LoggingCallback::on_epoch_begin(int epoch) {
        if (log_time_) {
            epoch_start_time_ = std::chrono::steady_clock::now();
        }
        // If trainer_ is set and has a get_max_epochs() method:
        // int max_epochs = trainer_ ? trainer_->get_max_epochs() : 0;
        // std::cout << prefix_ << " Epoch " << epoch << (max_epochs > 0 ? "/" + std::to_string(max_epochs) : "") << " started." << std::endl;
        std::cout << prefix_ << " Epoch " << epoch << " started." << std::endl;

    }

    void LoggingCallback::on_epoch_end(const EpochEndState& state) {
        std::cout << prefix_ << " Epoch " << state.epoch << " finished. ";
        if (log_time_) {
            auto end_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - epoch_start_time_);
            std::cout << "Time: " << duration.count() << "ms. ";
        }
        // Note: state.train_loss might be -1 if not explicitly set by Trainer in EpochEndState
        // The current Trainer example logs average train loss at the end of train_epoch directly.
        // If you want this callback to also log it, ensure EpochEndState.train_loss is populated.
        if (state.train_loss != -1.0) { // Assuming -1.0 is a sentinel for "not available"
             std::cout << "Avg Train Loss: " << std::fixed << std::setprecision(4) << state.train_loss << ". ";
        }
        if (state.val_loss != -1.0) {
            std::cout << "Avg Val Loss: " << std::fixed << std::setprecision(4) << state.val_loss << ".";
        }
        std::cout << std::endl;
    }

    void LoggingCallback::on_batch_begin(int epoch, int batch_idx) {
        // This can be too verbose, so often left empty or used for very specific logging.
        // if (log_time_) {
        //     batch_start_time_ = std::chrono::steady_clock::now();
        // }
        // std::cout << prefix_ << " E" << epoch << "/B" << batch_idx << " begin." << std::endl;
    }

    void LoggingCallback::on_batch_end(const BatchEndState& state) {
        // Log progress at specified intervals
        // Note: total_batches is not directly available in BatchEndState.
        // The trainer itself logs batch progress currently. This callback could do it too
        // if it had access to total_batches or if BatchEndState was augmented.
        if ((state.batch_idx + 1) % log_batch_interval_ == 0) {
            std::cout << prefix_ << " E" << state.epoch << " | Batch " << (state.batch_idx + 1)
                      << " | Loss: " << std::fixed << std::setprecision(4) << state.loss.item<double>()
                      << " (Samples: " << state.num_samples << ")" << std::endl;
        }
        // If you wanted per-batch timing (can be overhead):
        // if (log_time_) {
        //    auto end_time = std::chrono::steady_clock::now();
        //    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - batch_start_time_);
        //    std::cout << prefix_ << " E" << state.epoch << "/B" << state.batch_idx << " time: " << duration.count() << "us." << std::endl;
        // }
    }

} // namespace xt