#include <transforms/target/event_rate_calculator.h>

// #include "include/transforms/target/event_rate_calculator.h"
#include <stdexcept>

// Example Main - Uncomment to run a standalone test
/*
#include <iostream>

// Mock xt::Module for testing purposes
class xt::Module {
public:
    virtual ~Module() = default;
    virtual auto forward(std::initializer_list<std::any> tensors) -> std::any = 0;
};

void test_calculator(xt::transforms::target::EventRateCalculator& calculator,
                     const std::vector<double>& timestamps, double duration) {

    std::cout << "Input: " << timestamps.size() << " events over " << duration << " seconds." << std::endl;
    auto rate_any = calculator.forward({timestamps, duration});
    auto rate = std::any_cast<double>(rate_any);
    std::cout << "Output Rate: " << rate << std::endl;
}

int main() {
    // 1. --- Setup ---
    // A list of 5 events occurring within a 20-second window.
    std::vector<double> event_times = {1.2, 3.5, 4.0, 10.8, 19.1};
    double total_duration = 20.0;

    // --- Example 1: Calculate rate per second ---
    std::cout << "--- Calculating rate per SECOND ---" << std::endl;
    xt::transforms::target::EventRateCalculator calc_per_sec(1.0);
    // Expected: 5 events / 20.0s = 0.25 events/sec
    test_calculator(calc_per_sec, event_times, total_duration);

    // --- Example 2: Calculate rate per minute ---
    std::cout << "\n--- Calculating rate per MINUTE ---" << std::endl;
    xt::transforms::target::EventRateCalculator calc_per_min(60.0);
    // Expected: 0.25 events/sec * 60 s/min = 15 events/min
    test_calculator(calc_per_min, event_times, total_duration);

    // --- Example 3: Edge case with zero events ---
    std::cout << "\n--- Testing zero events ---" << std::endl;
    std::vector<double> no_events = {};
    // Expected: 0 events / 20.0s = 0 events/sec
    test_calculator(calc_per_sec, no_events, total_duration);

    // --- Test Error Handling ---
    std::cout << "\n--- Testing Error Handling (zero duration) ---" << std::endl;
    try {
        calc_per_sec.forward({event_times, 0.0});
    } catch(const std::invalid_argument& e) {
        std::cout << "Caught expected exception: " << e.what() << std::endl;
    }

    return 0;
}
*/

namespace xt::transforms::target {

    EventRateCalculator::EventRateCalculator(double per_time_unit)
        : per_time_unit_(per_time_unit) {

        if (per_time_unit_ <= 0.0) {
            throw std::invalid_argument("per_time_unit must be a positive value.");
        }
    }

    auto EventRateCalculator::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.size() != 2) {
            throw std::invalid_argument("EventRateCalculator::forward requires two inputs: {timestamps_vector, duration}.");
        }

        std::vector<double> timestamps;
        double duration;

        try {
            timestamps = std::any_cast<std::vector<double>>(any_vec[0]);
            duration = std::any_cast<double>(any_vec[1]);
        } catch (const std::bad_any_cast& e) {
            throw std::invalid_argument("Invalid input types. Expected std::vector<double> and double.");
        }

        // 2. --- Core Logic ---
        double num_events = static_cast<double>(timestamps.size());

        if (duration <= 0.0) {
            // If duration is zero (or negative), the rate is undefined if there are events.
            // If there are no events, the rate is zero.
            if (num_events > 0) {
                throw std::invalid_argument("Duration must be positive when there are one or more events.");
            }
            return 0.0;
        }

        double rate = (num_events / duration) * per_time_unit_;

        return rate;
    }

} // namespace xt::transforms::target