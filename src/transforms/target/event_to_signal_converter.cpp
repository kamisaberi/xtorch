#include <transforms/target/event_to_signal_converter.h>

#include <stdexcept>
#include <cmath> // For std::exp and M_PI

// Example Main - Uncomment to run a standalone test
/*
#include <iostream>

// Mock xt::Module for testing purposes
class xt::Module {
public:
    virtual ~Module() = default;
    virtual auto forward(std::initializer_list<std::any> tensors) -> std::any = 0;
};

int main() {
    // 1. --- Setup ---
    // We want to create a signal of length 50 that represents 10 seconds.
    double duration = 10.0;
    int samples = 50;

    // A list of events occurring within the 10-second window.
    std::vector<double> event_times = {2.0, 7.5};

    std::cout << "Signal: " << duration << "s duration, " << samples << " samples." << std::endl;
    std::cout << "Events at t=2.0s and t=7.5s." << std::endl;

    // --- Example 1: Using a DIRAC kernel ---
    std::cout << "\n--- Testing with DIRAC kernel ---" << std::endl;
    xt::transforms::target::EventToSignalConverter dirac_converter(duration, samples, xt::transforms::target::SignalKernel::DIRAC);
    auto dirac_any = dirac_converter.forward({event_times});
    auto dirac_signal = std::any_cast<torch::Tensor>(dirac_any);

    // Expected event indices:
    // Event 1: 2.0s * (50 samples / 10s) = index 10
    // Event 2: 7.5s * (50 samples / 10s) = index 37.5 -> round to 38
    std::cout << "Dirac Signal (showing non-zero values):" << std::endl;
    std::cout << "Value at index 10: " << dirac_signal[10].item<float>() << std::endl;
    std::cout << "Value at index 38: " << dirac_signal[38].item<float>() << std::endl;

    // --- Example 2: Using a GAUSSIAN kernel ---
    std::cout << "\n--- Testing with GAUSSIAN kernel ---" << std::endl;
    xt::transforms::target::EventToSignalConverter gauss_converter(duration, samples, xt::transforms::target::SignalKernel::GAUSSIAN, 2.0);
    auto gauss_any = gauss_converter.forward({event_times});
    auto gauss_signal = std::any_cast<torch::Tensor>(gauss_any);

    // Expected peaks at indices 10 and 38, with values spreading around them.
    std::cout << "Gaussian Signal (showing values around the first peak at index 10):" << std::endl;
    std::cout << "Index 9:  " << gauss_signal[9].item<float>() << std::endl;
    std::cout << "Index 10: " << gauss_signal[10].item<float>() << " (Peak)" << std::endl;
    std::cout << "Index 11: " << gauss_signal[11].item<float>() << std::endl;

    return 0;
}
*/

namespace xt::transforms::target {

    EventToSignalConverter::EventToSignalConverter(double signal_duration, int num_samples, SignalKernel kernel, double gauss_stddev)
        : signal_duration_(signal_duration), num_samples_(num_samples), kernel_(kernel), gauss_stddev_(gauss_stddev) {

        if (signal_duration_ <= 0.0) {
            throw std::invalid_argument("Signal duration must be positive.");
        }
        if (num_samples_ <= 0) {
            throw std::invalid_argument("Number of samples must be positive.");
        }
        if (kernel_ == SignalKernel::GAUSSIAN && gauss_stddev_ <= 0.0) {
            throw std::invalid_argument("Gaussian standard deviation must be positive.");
        }

        samples_per_unit_time_ = static_cast<double>(num_samples_) / signal_duration_;
    }

    auto EventToSignalConverter::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("EventToSignalConverter::forward received an empty list.");
        }

        std::vector<double> timestamps;
        try {
            timestamps = std::any_cast<std::vector<double>>(any_vec[0]);
        } catch (const std::bad_any_cast& e) {
            throw std::invalid_argument("Input to EventToSignalConverter must be a std::vector<double>.");
        }

        // 2. --- Signal Initialization ---
        torch::Tensor signal = torch::zeros({num_samples_}, torch::kFloat32);
        torch::Tensor sample_indices = torch::arange(0, num_samples_, torch::kFloat32);

        // 3. --- Place Kernels for Each Event ---
        for (const double event_time : timestamps) {
            if (event_time < 0.0 || event_time > signal_duration_) {
                continue; // Ignore events outside the signal duration
            }

            // Calculate the ideal, floating-point position of the event in the signal
            double event_center_idx = event_time * samples_per_unit_time_;

            if (kernel_ == SignalKernel::DIRAC) {
                // For dirac, just place a 1 at the nearest integer sample index.
                long nearest_idx = std::round(event_center_idx);
                if (nearest_idx >= 0 && nearest_idx < num_samples_) {
                    signal[nearest_idx] = 1.0f;
                }
            }
            else if (kernel_ == SignalKernel::GAUSSIAN) {
                // For gaussian, calculate the value of a gaussian function centered
                // at the event for every sample in the signal.
                // Formula: A * exp( -(x-mu)^2 / (2*sigma^2) )
                // Here, A=1, mu=event_center_idx, sigma=gauss_stddev_
                torch::Tensor exponent = -torch::pow(sample_indices - event_center_idx, 2) / (2 * std::pow(gauss_stddev_, 2));
                torch::Tensor gaussian_kernel = torch::exp(exponent);

                // Add the generated kernel to the signal. This allows multiple
                // overlapping kernels to sum up naturally.
                signal += gaussian_kernel;
            }
        }

        return signal;
    }

} // namespace xt::transforms::target