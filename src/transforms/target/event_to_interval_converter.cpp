#include <transforms/target/event_to_interval_converter.h>

#include <stdexcept>
#include <algorithm> // For std::is_sorted

// Example Main - Uncomment to run a standalone test
/*
#include <iostream>

// Mock xt::Module for testing purposes
class xt::Module {
public:
    virtual ~Module() = default;
    virtual auto forward(std::initializer_list<std::any> tensors) -> std::any = 0;
};

// Helper to print a vector
template<typename T>
void print_vector(const std::string& name, const std::vector<T>& vec) {
    std::cout << name << "[ ";
    for (const auto& val : vec) {
        std::cout << val << " ";
    }
    std::cout << "]" << std::endl;
}


int main() {
    // 1. --- Setup ---
    xt::transforms::target::EventToIntervalConverter converter;

    // --- Example 1: Standard case ---
    std::cout << "--- Testing standard case ---" << std::endl;
    std::vector<double> event_times = {2.5, 4.0, 7.5, 8.0, 10.0};
    print_vector("Input Timestamps:  ", event_times);

    auto intervals_any = converter.forward({event_times});
    auto intervals = std::any_cast<std::vector<double>>(intervals_any);

    // Expected output: [4.0-2.5, 7.5-4.0, 8.0-7.5, 10.0-8.0] -> [1.5, 3.5, 0.5, 2.0]
    print_vector("Output Intervals: ", intervals);

    // --- Example 2: Edge cases ---
    std::cout << "\n--- Testing edge cases ---" << std::endl;
    std::vector<double> single_event = {100.0};
    print_vector("Input (single event):", single_event);
    auto single_any = converter.forward({single_event});
    print_vector("Output (empty):      ", std::any_cast<std::vector<double>>(single_any));

    std::vector<double> no_events = {};
    print_vector("\nInput (no events):   ", no_events);
    auto none_any = converter.forward({no_events});
    print_vector("Output (empty):      ", std::any_cast<std::vector<double>>(none_any));

    // --- Test Error Handling ---
    std::cout << "\n--- Testing Error Handling (unsorted input) ---" << std::endl;
    try {
        std::vector<double> unsorted_times = {5.0, 2.0, 8.0};
        print_vector("Input (unsorted):    ", unsorted_times);
        converter.forward({unsorted_times});
    } catch(const std::invalid_argument& e) {
        std::cout << "Caught expected exception: " << e.what() << std::endl;
    }

    return 0;
}
*/

namespace xt::transforms::target {

    EventToIntervalConverter::EventToIntervalConverter() = default;

    auto EventToIntervalConverter::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("EventToIntervalConverter::forward received an empty list.");
        }

        std::vector<double> timestamps;
        try {
            timestamps = std::any_cast<std::vector<double>>(any_vec[0]);
        } catch (const std::bad_any_cast& e) {
            throw std::invalid_argument("Input to EventToIntervalConverter must be a std::vector<double>.");
        }

        // It is crucial that timestamps are sorted to calculate meaningful intervals.
        if (!std::is_sorted(timestamps.begin(), timestamps.end())) {
            throw std::invalid_argument("Input timestamps must be sorted in ascending order.");
        }

        // 2. --- Handle Edge Cases ---
        // If there are 0 or 1 events, no intervals can be formed.
        if (timestamps.size() <= 1) {
            return std::vector<double>{};
        }

        // 3. --- Core Logic ---
        std::vector<double> intervals;
        intervals.reserve(timestamps.size() - 1); // Pre-allocate memory for N-1 intervals

        for (size_t i = 1; i < timestamps.size(); ++i) {
            double interval = timestamps[i] - timestamps[i-1];
            intervals.push_back(interval);
        }

        return intervals;
    }

} // namespace xt::transforms::target